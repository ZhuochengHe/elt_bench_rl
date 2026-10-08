"Render ELT-Bench prompts for the selected destination."

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import jinja2
import yaml

from . import benchmark

SWE_ROOT = benchmark.REPO / "agents" / "SWE-agent"
PROMPT_CONFIG_PATH = SWE_ROOT / "config" / "elt_fc.yaml"


TEMPLATE_VARS: dict[str, str] = {"working_dir": "/workspace", "open_file": ""}


_SHELL_PROMPT_RE = re.compile(
    r"Your shell prompt is formatted as follows:\n"
    r"\(Open file: <path>\)\n"
    r"\(Current directory: <cwd>\)\n"
    r"bash-\$\n"
)

_FUNCTION_CALLING_BLOCK = """\
You interact through tool calls in the native function-calling format.
(Open file: <path>) / (Current directory: <cwd>) / bash-$ is only how a human would
see the shell; you do not need to print it.
"""


_STAGE1_RE = re.compile(r"#\s*Stage 1:.*?(?=#\s*Stage 2:)", re.S)

#: Tinker adapter exposes the same named file/search tools as the SWE-agent ELT preset;
#: guidance clarifies the arguments so models don't fall back to bash heredocs.
_TOOL_GUIDANCE = """\
# Tool usage
Use the SWE-agent file tools: `open` to view a file, `goto`/`scroll_up`/`scroll_down`
to navigate, `create` to create a file, `edit(search, replace, replace_all=false)`
to replace text in the current window, and `insert(text, line?)` to add text to the
current file (at EOF if line is omitted). The `text` argument is required for insert.
Use `find_file`, `search_dir`, and `search_file` to locate files/content. Use `bash`
for Terraform, dbt, and other shell commands, and `execute_sql` to query the warehouse.
Only create or edit files under `/workspace/elt`.
"""

_STAGE1_ALREADY_DONE = """\
# Stage 1: Extraction and Loading (already done)
The source tables are already extracted and loaded into the warehouse for you.
Do NOT configure Airbyte or Terraform and do NOT re-run extraction: go straight to
Stage 2. The source tables and their columns are described in /workspace/schemas,
and the connection details of the pre-loaded source are in /workspace/config.yaml.

"""


@dataclass
class Prompt:
    system: str
    instance: str

    def messages(self, renderer, tools: list) -> list:
        prefix = renderer.create_conversation_prefix_with_tools(
            tools=[t.to_spec() for t in tools], system_prompt=self.system
        )
        return prefix + [
            {"role": "user", "content": _TOOL_GUIDANCE + "\n" + self.instance}
        ]


@lru_cache(maxsize=8)
def _agent_config_cached(destination: str) -> str:
    import run_elt

    with tempfile.TemporaryDirectory(prefix="eltbench-prompt-") as tmp:
        out = Path(tmp) / "elt_fc.yaml"
        run_elt.write_destination_config(destination, out)
        return out.read_text(encoding="utf-8")


def load_agent_config(destination: str) -> dict:
    return yaml.safe_load(_agent_config_cached(destination)) or {}


def render_template(text: str, extra_vars: dict[str, str] | None = None) -> str:
    variables = {**TEMPLATE_VARS, **(extra_vars or {})}
    env = jinja2.Environment(undefined=jinja2.StrictUndefined, autoescape=False)
    try:
        return env.from_string(text).render(**variables)
    except jinja2.UndefinedError as exc:
        raise ValueError(
            f"The upstream prompt template contains unknown variables ({exc}); "
            f"add them to prompt.TEMPLATE_VARS: {sorted(variables)}"
        ) from exc


def to_function_calling(system: str) -> str:
    new, count = _SHELL_PROMPT_RE.subn(_FUNCTION_CALLING_BLOCK, system)
    if count != 1:
        raise ValueError(
            "Expected shell-prompt instructions were not found in the system prompt; "
            f"matched {count} sections. Update prompt._SHELL_PROMPT_RE."
        )
    return new


def drop_stage1(instance: str) -> str:
    new, count = _STAGE1_RE.subn(_STAGE1_ALREADY_DONE, instance)
    if count != 1:
        raise ValueError(
            f"Expected exactly one Stage 1 section in the instance template; found {count}. "
            "Update prompt._STAGE1_RE for the current upstream template."
        )
    return new


def load_prompt(destination: str, *, stage1: bool = True) -> Prompt:
    config = load_agent_config(destination)
    templates = (config.get("agent") or {}).get("templates") or {}
    system = templates.get("system_template") or ""
    instance = templates.get("instance_template") or ""
    if not system or not instance:
        raise ValueError(
            f"{PROMPT_CONFIG_PATH} is missing system_template or instance_template."
        )

    system = to_function_calling(render_template(system))
    instance = render_template(instance)
    if not stage1:
        instance = drop_stage1(instance)
    return Prompt(system=system, instance=instance)
