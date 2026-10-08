"Implement the Tinker tool environment and rollout lifecycle."

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shlex
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import chz
from tinker_cookbook.rl.rollout_presets import RolloutConfig, agentic
from tinker_cookbook.sandbox import SandboxResult
from tinker_cookbook.tool_use import build_agent_tool_env
from tinker_cookbook.tool_use.tools import simple_tool_result, tool
from tinker_cookbook.tool_use.types import Tool, ToolResult

from . import benchmark
from .destinations import make_reader
from .milestones import verify_milestones
from .prompt import Prompt, load_prompt
from .reward import RewardBreakdown, compute_reward
from .reward import source_snapshot as reward_source_snapshot
from .workspace import (
    RolloutWorkspace,
    attempt_tag,
    drop_namespace,
    keep_namespaces,
    materialize_workspace,
    new_run_tag,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = "elt-swe:local"
DEFAULT_NETWORK = "elt-docker_elt_network"

CONTAINER_WS = "/workspace"
SWE_TOOLS_HOST = ROOT / "repo/agents/SWE-agent/tools"
SWE_TOOLS_CONTAINER = "/opt/swe-agent-tools"


@dataclass
class ELTContainer:
    host_workspace: Path
    image: str = DEFAULT_IMAGE
    network: str = DEFAULT_NETWORK
    name: str = field(default_factory=lambda: f"elt-agent-{uuid.uuid4().hex[:8]}")

    def start(self) -> None:
        if not (SWE_TOOLS_HOST / "defaults/bin/open").is_file():
            raise FileNotFoundError(
                f"SWE-agent tool bundle not found: {SWE_TOOLS_HOST}"
            )
        self.host_workspace.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                self.name,
                "--network",
                self.network,
                "-v",
                f"{self.host_workspace.resolve()}:{CONTAINER_WS}",
                "-v",
                f"{SWE_TOOLS_HOST.resolve()}:{SWE_TOOLS_CONTAINER}:ro",
                "-w",
                CONTAINER_WS,
                self.image,
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    def freeze(self) -> None:
        subprocess.run(["docker", "pause", self.name], capture_output=True, text=True)

    def stop(self) -> None:

        self.chown_workspace()
        subprocess.run(
            ["docker", "rm", "-f", self.name], capture_output=True, text=True
        )

    def chown_workspace(self) -> None:
        if not hasattr(os, "getuid") or os.getuid() == 0:
            return
        try:
            subprocess.run(
                [
                    "docker",
                    "exec",
                    self.name,
                    "chown",
                    "-R",
                    f"{os.getuid()}:{os.getgid()}",
                    CONTAINER_WS,
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
        except Exception:  # noqa: BLE001
            pass

    def alive(self) -> bool:
        r = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", self.name],
            capture_output=True,
            text=True,
        )
        return r.returncode == 0 and r.stdout.strip() == "true"

    def exec(self, command: str, timeout: int = 600) -> tuple[int, str]:
        r = subprocess.run(
            ["docker", "exec", "-w", CONTAINER_WS, self.name, "bash", "-lc", command],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return r.returncode, (r.stdout or "") + (r.stderr or "")

    @property
    def sandbox_id(self) -> str:
        return self.name

    async def run_command(
        self,
        command: str,
        workdir: str | None = None,
        timeout: int = 60,
        max_output_bytes: int | None = None,
    ) -> SandboxResult:
        final = f"cd {workdir} && {command}" if workdir else command
        try:
            code, out = await asyncio.to_thread(self.exec, final, timeout)
        except subprocess.TimeoutExpired:
            return SandboxResult(
                stdout="", stderr=f"timeout after {timeout}s", exit_code=124
            )
        if max_output_bytes is not None and len(out) > max_output_bytes:
            out = out[:max_output_bytes]
        return SandboxResult(stdout=out, stderr="", exit_code=code)

    async def read_file(
        self, path: str, max_bytes: int | None = None, timeout: int = 60
    ) -> SandboxResult:
        rel = path.lstrip("/")
        if rel.startswith("workspace/"):
            rel = rel[len("workspace/") :]
        target = self.host_workspace / rel
        try:
            data = target.read_bytes()
        except OSError as exc:
            return SandboxResult(stdout="", stderr=str(exc), exit_code=1)
        if max_bytes is not None:
            data = data[:max_bytes]
        return SandboxResult(
            stdout=data.decode("utf-8", "replace"), stderr="", exit_code=0
        )

    async def write_file(
        self,
        path: str,
        content: str | bytes,
        executable: bool = False,
        timeout: int = 60,
    ) -> SandboxResult:
        rel = path.lstrip("/")
        if rel.startswith("workspace/"):
            rel = rel[len("workspace/") :]
        target = self.host_workspace / rel
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, bytes):
                target.write_bytes(content)
            else:
                target.write_text(content, encoding="utf-8")
            if executable:
                target.chmod(0o755)
        except OSError as exc:
            return SandboxResult(stdout="", stderr=str(exc), exit_code=1)
        return SandboxResult(stdout="", stderr="", exit_code=0)

    async def send_heartbeat(self, timeout: int = 30) -> None:
        return None

    async def cleanup(self) -> None:
        await asyncio.to_thread(self.stop)


@dataclass
class TaskContext:
    name: str
    destination: str
    task_dir: Path
    gt_dir: Path

    source_db: str = ""
    source_schema: str = "public"

    load_tables: list[str] = field(default_factory=list)

    @property
    def models(self) -> list[str]:
        return [p.stem for p in sorted(self.gt_dir.glob("*.csv"))]


def load_task(
    name: str,
    *,
    destination: str = benchmark.LOCAL_DESTINATION,
    load_tables: list[str] | None = None,
) -> TaskContext:
    task_dir = benchmark.benchmark_dir(destination) / name
    if not task_dir.is_dir():
        raise FileNotFoundError(f"Task directory does not exist: {task_dir}")
    gt_dir = benchmark.GT_ROOT / name
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"Ground-truth directory does not exist: {gt_dir}")
    return TaskContext(
        name=name,
        destination=destination,
        task_dir=task_dir,
        gt_dir=gt_dir,
        source_db=name,
        source_schema="public",
        load_tables=(
            benchmark.source_tables(name) if load_tables is None else list(load_tables)
        ),
    )


def _session_context_sql(destination: str, database: str, schema: str) -> list[str]:
    key = (destination or "").lower()
    if key == "snowflake":
        return [f'USE DATABASE "{database}"', f'USE SCHEMA "{schema}"']
    if key == "databricks":
        return [f"USE CATALOG `{database}`", f"USE SCHEMA `{schema}`"]
    if key in ("redshift", benchmark.LOCAL_DESTINATION):
        return [f'SET search_path TO "{schema}"']
    return []


def _readonly_context_sql(destination: str) -> list[str]:
    if (destination or "").lower() in ("redshift", benchmark.LOCAL_DESTINATION):
        return ["SET default_transaction_read_only = on"]
    return []


def _database_allowed_error(db: str, namespace, source_db: str) -> str | None:
    allowed = {str(namespace.eval_database).lower()}
    if source_db:
        allowed.add(str(source_db).lower())
    if str(db).lower() not in allowed:
        return (
            f"Denied: database={db!r} is outside the task allowlist "
            f"(target={namespace.eval_database}; source={source_db or 'none'})."
        )
    return None


class ELTTools:
    WRITABLE = ("elt",)

    def __init__(
        self,
        container: ELTContainer,
        task: TaskContext,
        workspace: RolloutWorkspace,
        destination: str,
        credential_path: Path | None = None,
    ):
        self.container = container
        self.task = task
        self.workspace = workspace
        self.destination = destination
        self.credential_path = credential_path

        self.namespace = workspace.namespace
        self.submitted = False
        self.last_command = ""
        self.editor_path: Path | None = None

    def _resolve(self, path: str) -> Path:
        rel = path.lstrip("/")
        if rel.startswith("workspace/"):
            rel = rel[len("workspace/") :]
        target = (self.container.host_workspace / rel).resolve()
        ws = self.container.host_workspace.resolve()
        if ws != target and ws not in target.parents:
            raise ValueError(f"Path is outside the workspace: {path}")
        return target

    # ------------------------------------------------------------ shell
    @tool
    async def bash(self, command: str) -> ToolResult:
        """Execute a shell command inside the workspace container.

        Args:
            command: The shell command to run (bash -lc).
        """
        self.last_command = command
        try:
            code, out = await asyncio.to_thread(self.container.exec, command)
        except subprocess.TimeoutExpired:
            return simple_tool_result("Command timed out (600 seconds).")
        return simple_tool_result(out or f"(no output, exit={code})")

    # -------------------------------------------- SWE-agent function-calling adapters
    def _workspace_path(self, path: str, *, writable: bool = False) -> tuple[Path, str]:
        target = self._resolve(path)
        root = self.container.host_workspace.resolve()
        if writable:
            elt_root = (root / "elt").resolve()
            if target != elt_root and elt_root not in target.parents:
                raise ValueError("File edits are restricted to /workspace/elt")
        relative = target.relative_to(root)
        return target, f"{CONTAINER_WS}/{relative.as_posix()}"

    async def _run_swe_tool(
        self, name: str, args: list[str], *, shell: bool = False
    ) -> ToolResult:
        """Invoke the checked-in SWE-agent bundle inside this rollout's container."""
        scripts = {
            "open": "defaults/bin/open",
            "goto": "defaults/bin/goto",
            "scroll_up": "defaults/bin/scroll_up",
            "scroll_down": "defaults/bin/scroll_down",
            "create": "defaults/bin/create",
            "edit": "edit_replace/bin/edit",
            "insert": "edit_replace/bin/insert",
            "find_file": "search/bin/find_file",
            "search_dir": "search/bin/search_dir",
            "search_file": "search/bin/search_file",
        }
        script = f"{SWE_TOOLS_CONTAINER}/{scripts[name]}"
        library_path = (
            f"{SWE_TOOLS_CONTAINER}/defaults/lib:{SWE_TOOLS_CONTAINER}/registry/lib"
        )
        binary_path = ":".join(
            f"{SWE_TOOLS_CONTAINER}/{part}"
            for part in (
                "defaults/bin",
                "registry/bin",
                "search/bin",
                "edit_replace/bin",
            )
        )
        registry_file = f"/tmp/elt-swe-{self.container.name}.json"
        env = (
            f"export PYTHONPATH={shlex.quote(library_path)}; "
            "export WINDOW=100; export OVERLAP=2; "
            f"export PATH={shlex.quote(binary_path)}:$PATH; "
            f"export SWE_AGENT_ENV_FILE={shlex.quote(registry_file)}; "
        )
        runner = "bash" if shell else "python"
        command = env + shlex.join([runner, script, *args])
        try:
            code, out = await asyncio.to_thread(self.container.exec, command)
        except subprocess.TimeoutExpired:
            return simple_tool_result("SWE-agent tool timed out (600s)")
        return simple_tool_result(out.strip() or f"(no output, exit={code})")

    def _require_editable_open_file(self) -> None:
        if self.editor_path is None:
            raise ValueError("No file open. Use `open` or `create` first.")
        relative = self.editor_path.relative_to(self.container.host_workspace.resolve())
        self._workspace_path(relative.as_posix(), writable=True)
        if not self.editor_path.is_file():
            raise ValueError(
                "The current open path is not a file and cannot be edited."
            )

    @tool
    async def open(self, path: str = "", line_number: int | None = None) -> ToolResult:
        """Open a workspace file or directory, optionally at a 1-based line number."""
        args: list[str] = []
        if path:
            try:
                target, container_path = self._workspace_path(path)
            except ValueError as exc:
                return simple_tool_result(str(exc))
            args.append(container_path)
            if target.is_file():
                self.editor_path = target
        elif self.editor_path is None:
            return simple_tool_result('Usage: open "<file>"')
        if line_number is not None:
            args.append(str(line_number))
        return await self._run_swe_tool("open", args)

    @tool
    async def goto(self, line_number: int) -> ToolResult:
        """Move the open-file window to a 1-based line number."""
        if self.editor_path is None:
            return simple_tool_result("No file open. Use `open` first.")
        return await self._run_swe_tool("goto", [str(line_number)])

    @tool
    async def scroll_up(self) -> ToolResult:
        """Scroll the open-file window up."""
        if self.editor_path is None:
            return simple_tool_result("No file open. Use `open` first.")
        return await self._run_swe_tool("scroll_up", [])

    @tool
    async def scroll_down(self) -> ToolResult:
        """Scroll the open-file window down."""
        if self.editor_path is None:
            return simple_tool_result("No file open. Use `open` first.")
        return await self._run_swe_tool("scroll_down", [])

    @tool
    async def create(self, filename: str) -> ToolResult:
        """Create and open a new file under /workspace/elt."""
        try:
            target, container_path = self._workspace_path(filename, writable=True)
        except ValueError as exc:
            return simple_tool_result(str(exc))
        if target.exists():
            return simple_tool_result(
                f"File already exists at: {container_path}. Cannot overwrite using `create`."
            )
        self.editor_path = target
        return await self._run_swe_tool("create", [container_path])

    @tool
    async def edit(
        self, search: str, replace: str, replace_all: bool = False
    ) -> ToolResult:
        """Replace text in the open-file window, or throughout the file when replace_all is true."""
        try:
            self._require_editable_open_file()
        except ValueError as exc:
            return simple_tool_result(str(exc))
        args = [search, replace]
        if replace_all:
            args.append("true")
        return await self._run_swe_tool("edit", args)

    @tool
    async def insert(self, text: str, line: int | None = None) -> ToolResult:
        """Insert text at EOF or after the optional 1-based line in the open file."""
        try:
            self._require_editable_open_file()
        except ValueError as exc:
            return simple_tool_result(str(exc))
        if not text:
            return simple_tool_result(
                "No text supplied; file was not modified. `text` is required."
            )
        args = [text]
        if line is not None:
            args.append(str(line))
        return await self._run_swe_tool("insert", args)

    @tool
    async def find_file(self, file_name: str, dir: str = ".") -> ToolResult:
        """Find files by name/pattern under a workspace directory."""
        try:
            _, container_dir = self._workspace_path(dir)
        except ValueError as exc:
            return simple_tool_result(str(exc))
        return await self._run_swe_tool(
            "find_file", [file_name, container_dir], shell=True
        )

    @tool
    async def search_dir(self, search_term: str, dir: str = ".") -> ToolResult:
        """Search file contents under a workspace directory."""
        try:
            _, container_dir = self._workspace_path(dir)
        except ValueError as exc:
            return simple_tool_result(str(exc))
        return await self._run_swe_tool(
            "search_dir", [search_term, container_dir], shell=True
        )

    @tool
    async def search_file(self, search_term: str, file: str = "") -> ToolResult:
        """Search a file, defaulting to the currently open file."""
        args = [search_term]
        if file or self.editor_path is not None:
            try:
                target = (
                    file
                    or self.editor_path.relative_to(
                        self.container.host_workspace.resolve()
                    ).as_posix()
                )
                _, container_file = self._workspace_path(target)
            except ValueError as exc:
                return simple_tool_result(str(exc))
            args.append(container_file)
        else:
            return simple_tool_result("No file open. Use `open` first or pass `file`.")
        return await self._run_swe_tool("search_file", args, shell=True)

    # ------------------------------------------------------------ SQL
    @tool
    async def execute_sql(
        self, sql: str, database: str = "", target_schema: str = ""
    ) -> ToolResult:
        """Run a SQL statement against the destination warehouse and return the rows.

        The session starts in this task's destination (database, schema) — the
        pair named in /workspace/config.yaml — so unqualified table names there
        refer to your own destination schema. You may also pass `database` to
        READ the pre-loaded source database; writes there are refused (it is
        shared with the other rollouts of this task). Use this to create or
        verify the data models you build.

        Args:
            sql: The SQL statement to execute.
            database: Target database name (defaults to the task's destination).
            target_schema: Target schema name (defaults to the task's destination).
        """
        ns = self.namespace
        db = database or ns.eval_database
        target_schema = target_schema or (ns.physical_schema or ns.eval_schema)

        denied = _database_allowed_error(db, ns, self.task.source_db)
        if denied:
            return simple_tool_result(denied)
        is_source = str(db).lower() != str(ns.eval_database).lower()
        try:
            cfg = benchmark.connector_config(
                self.destination,
                workspace_config=self.workspace.harness_config,
                credential_path=self.credential_path,
            )

            ws_block = self.workspace.destination_config or {}
            for key in ("user", "password"):
                if ws_block.get(key):
                    cfg[key] = ws_block[key]
            cfg["database"] = db
            reader = make_reader(
                self.destination, cfg, database=db, schema=target_schema
            )
        except Exception as exc:  # noqa: BLE001
            return simple_tool_result(f"Connection failed: {type(exc).__name__}: {exc}")
        try:
            conn = reader.connector if hasattr(reader, "connector") else reader
            raw = conn.conn()
            cur = raw.cursor()
            try:
                for stmt in _session_context_sql(self.destination, db, target_schema):
                    cur.execute(stmt)
                if is_source:
                    for stmt in _readonly_context_sql(self.destination):
                        cur.execute(stmt)
                cur.execute(sql)
                if cur.description:
                    cols = [d[0] for d in cur.description]
                    rows = cur.fetchmany(50)
                    body = " | ".join(str(c) for c in cols)
                    body = body + (
                        "\n" + "\n".join(" | ".join(str(v) for v in r) for r in rows)
                        if rows
                        else "\n(no rows)"
                    )
                elif cur.rowcount and cur.rowcount > 0:
                    body = f"Execution succeeded ({cur.rowcount} rows affected)."
                else:
                    body = "Execution succeeded (DDL or no row count)."
            finally:
                cur.close()
        except Exception as exc:  # noqa: BLE001
            return simple_tool_result(
                f"SQL failed: {type(exc).__name__}: {exc}\n"
                f"(current target: {db}.{target_schema})"
            )
        finally:
            try:
                reader.close()
            except Exception:  # noqa: BLE001
                pass
        return simple_tool_result(f"[Target {db}.{target_schema}] {body}")

    @tool
    async def submit(self) -> ToolResult:
        """Finish the episode. Call this when the pipeline is complete and verified."""
        self.submitted = True
        if os.environ.get("ELT_FREEZE_ON_SUBMIT", "1") not in ("0", "false", "no"):
            try:
                await asyncio.to_thread(self.container.freeze)
            except Exception:  # noqa: BLE001
                pass
        return simple_tool_result(
            "Submitted. The episode is complete.", should_stop=True
        )

    def all(self) -> list[Tool]:
        # SWE-agent elt_fc bundle, adapted as Tinker function tools, plus ELT-specific tools.
        return [
            self.bash,
            self.open,
            self.goto,
            self.scroll_up,
            self.scroll_down,
            self.create,
            self.edit,
            self.insert,
            self.find_file,
            self.search_dir,
            self.search_file,
            self.execute_sql,
            self.submit,
        ]


_READ_CMD = re.compile(r"\b(cat|less|more|head|tail|sed|awk|grep)\b")

_WRITE_CMD = re.compile(
    r"(>>?|\brm\b|\bmkdir\b|\btouch\b|\bcp\b|\bmv\b|\bdbt\s+init\b|\bgit\s+init\b|\bterraform\s+(init|apply|plan)\b)"
)
_DBT_RUN = re.compile(r"\bdbt\s+(run|build)\b")


_DOC_FILE_HINT = re.compile(
    r"(?:documentation/[^/\s]+|schemas/[^/\s]+|(?:^|[\s/])config\.ya?ml(?:$|[\s]))",
    re.I,
)
_DOC_DIR_HINT = re.compile(r"(?:^|/)documentation(?:/|$)|(?:^|/)schemas(?:/|$)", re.I)


@dataclass
class ProcessSignals:
    score: float
    read_before_write: bool
    read_before_run: bool
    dbt_used: bool
    metrics: dict[str, float]


def process_signals(history) -> ProcessSignals:
    first_write: int | None = None
    first_run: int | None = None
    reads: list[int] = []
    dbt_used = False
    current_file = ""
    for i, m in enumerate(history):
        for c in m.get("tool_calls") or []:
            fn = getattr(c, "function", None)
            name = str(getattr(fn, "name", "") or "")
            try:
                args = json.loads(getattr(fn, "arguments", "") or "{}")
            except Exception:  # noqa: BLE001
                args = {}
            text = " ".join(str(args.get(k, "")) for k in ("path", "command"))
            if name == "open":
                current_file = str(args.get("path", "") or current_file)
                if _DOC_FILE_HINT.search(current_file):
                    reads.append(i)
            elif name == "search_file":
                searched_file = str(args.get("file", "") or current_file)
                if _DOC_FILE_HINT.search(searched_file):
                    reads.append(i)
            elif name == "search_dir":
                if _DOC_DIR_HINT.search(str(args.get("dir", ""))):
                    reads.append(i)
            elif name == "str_replace_editor":
                if str(args.get("command", "")) == "view":
                    if _DOC_FILE_HINT.search(str(args.get("path", ""))):
                        reads.append(i)
                else:
                    first_write = i if first_write is None else first_write
            elif name in {"create", "edit", "insert"}:
                first_write = i if first_write is None else first_write
            elif name == "bash":
                is_read = bool(_READ_CMD.search(text)) and bool(
                    _DOC_FILE_HINT.search(text)
                )
                if is_read:
                    reads.append(i)
                if _DBT_RUN.search(text):
                    dbt_used = True
                    first_run = i if first_run is None else first_run
                elif _WRITE_CMD.search(text) and not is_read:
                    first_write = i if first_write is None else first_write

    read_at = min(reads) if reads else None
    before_write = read_at is not None and (
        first_write is None or read_at < first_write
    )
    before_run = read_at is not None and first_run is not None and read_at < first_run

    if before_write:
        score = 0.2
    elif before_run:
        score = 0.1
    else:
        score = 0.0
    return ProcessSignals(
        score=score,
        read_before_write=bool(before_write),
        read_before_run=bool(before_run),
        dbt_used=dbt_used,
        metrics={
            "trajectory/process_score": score,
            "trajectory/read_before_write": float(before_write),
            "trajectory/read_before_run": float(before_run),
            "trajectory/provenance_dbt": float(dbt_used),
        },
    )


def _verification_metric(history) -> dict[str, float]:
    for m in history:
        calls = m.get("tool_calls") or []
        if not calls:
            continue
        fn = getattr(calls[0], "function", None)
        name = str(getattr(fn, "name", "") or "")
        if name == "execute_sql":
            return {"trajectory/verified": 1.0}
        if name == "bash" and "select" in str(m.get("content") or "").lower():
            return {"trajectory/verified": 1.0}
    return {"trajectory/verified": 0.0}


# ------------------------------------------------------------------ Env
def fresh_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)


def renderer_for(model_name: str, tokenizer, renderer_name: str | None = None):
    if renderer_name:
        from tinker_cookbook.renderers import get_renderer

        return get_renderer(renderer_name, tokenizer)
    from tinker_cookbook.model_info import get_recommended_renderer_name
    from tinker_cookbook.renderers import get_renderer

    return get_renderer(get_recommended_renderer_name(model_name), tokenizer)


@dataclass
class ELTEnvConfig:
    task: TaskContext
    model_name: str
    destination: str = benchmark.LOCAL_DESTINATION
    group_size: int = 4
    max_turns: int = 50

    max_trajectory_tokens: int = 30000

    process_weight: float = 1.0

    provenance_check: bool = False

    max_tool_calls: int = 60

    rollout_timeout_seconds: float | None = None

    renderer_name: str | None = None
    image: str = DEFAULT_IMAGE
    network: str = DEFAULT_NETWORK

    base_destination_config: dict = field(default_factory=dict)

    harness_destination_config: dict = field(default_factory=dict)

    credential_path: Path | None = None
    runs_root: Path = field(default_factory=lambda: ROOT / "runs")

    staging_root: Path | None = None

    run_tag: str = ""

    load_mode: str = "auto"

    def __post_init__(self) -> None:
        if self.task.destination and self.destination != self.task.destination:
            raise ValueError(
                f"ELTEnvConfig.destination={self.destination!r} does not match task context "
                f"{self.task.destination!r}; both must use the same backend."
            )

    @property
    def effective_load_mode(self) -> str:
        if self.load_mode in ("content", "count"):
            return self.load_mode
        return benchmark.load_mode_for(self.destination)

    @property
    def stage1(self) -> bool:
        return bool(self.task.load_tables)

    @property
    def effective_staging_root(self) -> Path:
        return (
            Path(self.staging_root)
            if self.staging_root
            else Path(self.runs_root) / "_inputs" / self.destination
        )


def build_rollout_config(cfg: ELTEnvConfig) -> RolloutConfig:
    base = agentic()
    limits = chz.replace(
        base.limits,
        max_turns=cfg.max_turns,
        max_trajectory_tokens=cfg.max_trajectory_tokens,
        max_tool_calls=cfg.max_tool_calls,
        rollout_timeout_seconds=cfg.rollout_timeout_seconds,
    )

    return chz.replace(base, limits=limits, termination=rollout_termination())


def rollout_termination():

    return chz.replace(agentic().termination, zero_reward_on_limit=False)


class ELTEnvGroupBuilder:
    def __init__(self, cfg: ELTEnvConfig, tokenizer, renderer=None):
        self.cfg = cfg
        self.tokenizer = tokenizer

        self.renderer = renderer or renderer_for(
            cfg.model_name, tokenizer, cfg.renderer_name
        )
        self.rollout_config = build_rollout_config(cfg)
        self.run_tag = cfg.run_tag or new_run_tag()
        self.run_dir = Path(cfg.runs_root) / f"{cfg.task.name}-{self.run_tag}"

        self.rollouts: list[
            tuple[RolloutWorkspace, ELTContainer, object, ELTTools]
        ] = []

        self.attempts = 0

        self._source_hashes: dict[str, str] | None = None
        self.prompt: Prompt = load_prompt(cfg.destination, stage1=cfg.stage1)

    async def compute_group_rewards(self, trajectory_group, env_group):
        del env_group
        out: list[tuple[float, dict[str, float]]] = []
        for traj in trajectory_group:
            reward = 0.0
            metrics: dict[str, float] = {}
            transitions = list(getattr(traj, "transitions", []) or [])
            if transitions:
                last = transitions[-1]
                reward = float(getattr(last, "reward", 0.0) or 0.0)
                raw = (
                    getattr(last, "logs", None) or getattr(last, "metrics", None) or {}
                )
                metrics = {
                    k: float(v)
                    for k, v in dict(raw).items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                }
                if not raw:
                    reward = float(
                        sum(
                            float(getattr(t, "reward", 0.0) or 0.0) for t in transitions
                        )
                    )
            out.append((reward, metrics))
        return out

    async def make_envs(self):
        cfg = self.cfg
        task = cfg.task
        self.attempts += 1
        attempt_dir = self.run_dir / f"a{self.attempts}"
        attempt_dir.mkdir(parents=True, exist_ok=True)

        attempt_run_tag = attempt_tag(self.run_tag, self.attempts)
        envs = []
        for i in range(cfg.group_size):
            try:
                ws = await asyncio.to_thread(
                    materialize_workspace,
                    task.name,
                    attempt_dir / f"r{i}",
                    destination=cfg.destination,
                    index=i,
                    run_tag=attempt_run_tag,
                    staging_root=cfg.effective_staging_root,
                    base_destination_config=cfg.base_destination_config or None,
                    harness_config=cfg.harness_destination_config or None,
                    credential_path=cfg.credential_path,
                )
            except Exception:
                await self.cleanup()
                raise

            container = ELTContainer(ws.root, image=cfg.image, network=cfg.network)
            await asyncio.to_thread(container.start)

            reader = make_reader(
                cfg.destination,
                benchmark.connector_config(
                    cfg.destination,
                    workspace_config=ws.harness_config,
                    credential_path=cfg.credential_path,
                ),
                database=ws.namespace.eval_database,
                schema=ws.namespace.eval_schema,
            )

            tools = ELTTools(
                container,
                task,
                ws,
                cfg.destination,
                credential_path=cfg.credential_path,
            )
            tool_list = tools.all()

            if (
                cfg.stage1
                and cfg.effective_load_mode == "content"
                and self._source_hashes is None
            ):
                self._source_hashes = await asyncio.to_thread(
                    reward_source_snapshot,
                    reader,
                    task.source_db,
                    task.source_schema,
                    list(task.load_tables),
                )
            self.rollouts.append((ws, container, reader, tools))

            env_renderer = renderer_for(
                cfg.model_name, fresh_tokenizer(cfg.model_name), cfg.renderer_name
            )
            envs.append(
                build_agent_tool_env(
                    renderer=env_renderer,
                    tools=tool_list,
                    initial_messages=self.prompt.messages(self.renderer, tool_list),
                    reward_fn=self._make_reward(ws, reader),
                    max_turns=cfg.max_turns,
                    rollout_config=self.rollout_config,
                )
            )
        return envs

    def _make_reward(self, ws: RolloutWorkspace, reader):
        cfg = self.cfg
        ns = ws.namespace

        async def reward_fn(history) -> tuple[float, dict[str, float]]:
            def grade() -> RewardBreakdown:
                return compute_reward(
                    reader,
                    cfg.task.name,
                    cfg.task.gt_dir,
                    ns.eval_database,
                    ns.eval_schema,
                    source_db=cfg.task.source_db if cfg.task.load_tables else None,
                    source_schema=cfg.task.source_schema,
                    load_tables=cfg.task.load_tables,
                    load_mode=cfg.effective_load_mode,
                    source_hashes=self._source_hashes,
                )

            try:
                b = await asyncio.to_thread(grade)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Grading failed: task=%s rollout=%s target=%s.%s",
                    cfg.task.name,
                    ns.label,
                    ns.eval_database,
                    ns.eval_schema,
                )
                return 0.0, {"reward/error": 1.0, "reward/episode_total": 0.0}
            metrics = b.metrics()
            metrics.update(_verification_metric(history))
            proc = process_signals(history)
            milestones = await asyncio.to_thread(
                verify_milestones,
                workspace=ws.root,
                target_database=ns.eval_database,
                credentials_path=(
                    ROOT / ".secrets" / "airbyte_app.json"
                    if cfg.destination != benchmark.LOCAL_DESTINATION
                    else None
                ),
                graded_models=b.models,
                load_score=b.load_score,
                load_evaluated=b.load_evaluated,
            )
            metrics.update(proc.metrics)
            metrics.update(milestones.metrics())

            total = b.total + cfg.process_weight * (proc.score + milestones.score)
            if cfg.provenance_check and not proc.dbt_used:
                metrics["reward/provenance_violation"] = 1.0
                total = min(total, 0.0)
            metrics["reward/outcome_total"] = b.total
            metrics["reward/episode_total"] = total
            logger.info(
                "Graded: task=%s rollout=%s %s | documentation=%.2f milestones=%.2f "
                "(read_before_write=%s, dbt=%s)",
                cfg.task.name,
                ns.label,
                b.summary(),
                proc.score,
                milestones.score,
                proc.read_before_write,
                proc.dbt_used,
            )
            return total, metrics

        return reward_fn

    async def cleanup(self) -> None:
        for ws, container, reader, _tools in self.rollouts:
            try:
                await asyncio.to_thread(container.stop)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Failed to stop container: %s", container.name, exc_info=True
                )
            try:
                reader.close()
            except Exception:  # noqa: BLE001
                pass
            if not keep_namespaces():
                try:
                    await asyncio.to_thread(
                        drop_namespace,
                        self.cfg.destination,
                        ws.harness_config,
                        ws.namespace,
                    )
                except Exception:  # noqa: BLE001
                    logger.warning(
                        "Failed to clean up namespace: %s",
                        ws.namespace.label,
                        exc_info=True,
                    )
        self.rollouts.clear()

    def logging_tags(self) -> list[str]:
        return [self.cfg.task.name, "elt"]
