"""Test rollout namespace isolation and environment invariants."""

from __future__ import annotations

import asyncio
import inspect
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eltbench import benchmark  # noqa: E402
from eltbench.env import (  # noqa: E402
    ELTEnvConfig,
    ELTEnvGroupBuilder,
    build_rollout_config,
    load_task,
    rollout_termination,
)
from eltbench.prompt import load_prompt  # noqa: E402
from eltbench.reward import compute_reward  # noqa: E402
from eltbench.workspace import materialize_workspace, plan_namespace  # noqa: E402

TASK = "address"
MODEL = "states"
RUN_TAG = "t3st01"


# ------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def scratch(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("eltbench-ws")


@pytest.fixture(scope="module")
def staging(scratch: Path) -> Path:
    return scratch / "_inputs"


def _local_base():
    return {
        "host": "elt-postgres",
        "port": 5432,
        "user": "postgres",
        "password": "testelt",
        "database": "elt_warehouse",
        "schema": "eltbench",
    }


def _sf_base():
    return {
        "database": "address",
        "schema": "AIRBYTE_SCHEMA",
        "account": "acct",
        "user": "u",
        "password": "p",
    }


def test_indices_get_distinct_namespaces():
    seen = set()
    for i in range(4):
        ns, _block = plan_namespace(
            benchmark.LOCAL_DESTINATION, TASK, _local_base(), i, RUN_TAG
        )
        seen.add((ns.eval_database, ns.eval_schema))
    assert len(seen) == 4, f"Namespace collision: {seen}"


def test_plan_is_deterministic():
    a = plan_namespace(benchmark.LOCAL_DESTINATION, TASK, _local_base(), 1, RUN_TAG)
    b = plan_namespace(benchmark.LOCAL_DESTINATION, TASK, _local_base(), 1, RUN_TAG)
    assert a == b


def test_tags_separate_different_runs():
    a, _ = plan_namespace(benchmark.LOCAL_DESTINATION, TASK, _local_base(), 0, "aaaaaa")
    b, _ = plan_namespace(benchmark.LOCAL_DESTINATION, TASK, _local_base(), 0, "bbbbbb")
    assert a.eval_schema != b.eval_schema


def test_snowflake_isolates_on_database():
    ns, block = plan_namespace("snowflake", TASK, _sf_base(), 2, RUN_TAG)
    assert block["schema"] == "AIRBYTE_SCHEMA"
    assert block["database"] == ns.eval_database
    assert ns.eval_schema == ns.eval_database
    assert ns.eval_database != "address"


def test_snowflake_database_name_is_unquoted_resolvable():
    for index in (0, 1):
        ns, block = plan_namespace("snowflake", TASK, _sf_base(), index, RUN_TAG)
        for name in (ns.eval_database, ns.eval_schema, block["database"]):
            assert name == name.upper(), (
                f"Snowflake database names must be uppercase; got {name!r}"
            )


@pytest.mark.parametrize(
    "destination", ["databricks", "redshift", benchmark.LOCAL_DESTINATION]
)
def test_other_backends_isolate_on_schema(destination):
    base = _sf_base() if destination != benchmark.LOCAL_DESTINATION else _local_base()
    ns, block = plan_namespace(destination, TASK, base, 2, RUN_TAG)
    assert block["schema"] == ns.eval_schema
    assert block["schema"] != base["schema"]
    assert block["database"] == base["database"]


@pytest.mark.parametrize(
    "destination", [benchmark.LOCAL_DESTINATION, "snowflake", "databricks", "redshift"]
)
def test_namespaces_are_distinct_across_tasks(destination):
    base = _local_base() if destination == benchmark.LOCAL_DESTINATION else _sf_base()
    seen: dict[tuple[str, str], str] = {}
    for task in ("address", "airline", "app_store"):
        ns, block = plan_namespace(destination, task, base, 0, RUN_TAG)
        key = (ns.eval_database, ns.eval_schema)
        assert key not in seen, f"Tasks {task} and {seen[key]} share namespace {key}"
        seen[key] = task

        assert block["schema"] == (ns.physical_schema or ns.eval_schema)


def test_retry_gets_fresh_namespace():
    from eltbench import env as env_module
    from eltbench.workspace import attempt_tag

    tags = [attempt_tag(RUN_TAG, n) for n in (1, 2)]
    assert len(set(tags)) == 2
    names = set()
    for t in tags:
        for destination in (benchmark.LOCAL_DESTINATION, "snowflake"):
            base = (
                _local_base()
                if destination == benchmark.LOCAL_DESTINATION
                else _sf_base()
            )
            ns, _ = plan_namespace(destination, TASK, base, 0, t)
            names.add((destination, ns.eval_database, ns.eval_schema))
    assert len(names) == len(tags) * 2, f"Namespace collision across attempts: {names}"

    source = inspect.getsource(env_module.ELTEnvGroupBuilder.make_envs)
    assert "attempt_tag(" in source, "make_envs must use attempt_tag to isolate retries"
    assert "run_tag=attempt_run_tag" in source, (
        "make_envs must pass the attempt-specific run tag"
    )


@pytest.mark.parametrize("destination", [benchmark.LOCAL_DESTINATION, "snowflake"])
def test_workspace_config_points_at_its_own_namespace(destination, scratch, staging):
    base = _local_base() if destination == benchmark.LOCAL_DESTINATION else _sf_base()
    ws = materialize_workspace(
        TASK,
        scratch / f"ws-{destination}",
        destination=destination,
        index=1,
        run_tag=RUN_TAG,
        staging_root=staging,
        base_destination_config=base,
        harness_config={"host": "localhost", "port": 5433},
        prepare=False,
    )

    config = yaml.safe_load((ws.root / "config.yaml").read_text(encoding="utf-8"))
    block = config[destination]["config"]
    assert block["database"] == ws.namespace.eval_database
    if destination == "snowflake":
        assert block["schema"] == "AIRBYTE_SCHEMA"
        assert ws.namespace.eval_schema == ws.namespace.eval_database
    else:
        assert block["schema"] == ws.namespace.eval_schema

    for name in (
        "config.yaml",
        "data_model.yaml",
        "schemas",
        "documentation",
        "check_job_status.py",
        "elt",
    ):
        assert (ws.root / name).exists(), f"Workspace is missing {name}"
    assert (ws.root / "elt" / "main.tf").is_file()

    assert not list(ws.root.glob("*_credential.json"))


def test_fixture_destination_block_is_removed(scratch, staging):
    ws = materialize_workspace(
        TASK,
        scratch / "ws-fixture",
        destination=benchmark.LOCAL_DESTINATION,
        index=0,
        run_tag=RUN_TAG,
        staging_root=staging,
        base_destination_config=_local_base(),
        prepare=False,
    )
    config = yaml.safe_load((ws.root / "config.yaml").read_text(encoding="utf-8"))
    assert "snowflake" not in config
    assert benchmark.LOCAL_DESTINATION in config

    assert config["postgres"]["config"]["database"] == TASK


def test_harness_view_differs_only_in_endpoint(scratch, staging):
    ws = materialize_workspace(
        TASK,
        scratch / "ws-view",
        destination=benchmark.LOCAL_DESTINATION,
        index=3,
        run_tag=RUN_TAG,
        staging_root=staging,
        base_destination_config=_local_base(),
        harness_config={"host": "localhost", "port": 5433},
        prepare=False,
    )
    agent, harness = ws.destination_config, ws.harness_config
    assert agent["host"] == "elt-postgres" and agent["port"] == 5432
    assert harness["host"] == "localhost" and harness["port"] == 5433

    for key in ("database", "schema", "user", "password"):
        assert agent[key] == harness[key], key


def test_task_uses_official_benchmark_and_manifest():
    task = load_task(TASK, destination=benchmark.LOCAL_DESTINATION, load_tables=None)
    assert task.task_dir == benchmark.benchmark_dir(benchmark.LOCAL_DESTINATION) / TASK
    assert task.gt_dir == benchmark.GT_ROOT / TASK
    assert task.load_tables == benchmark.source_tables(TASK)
    assert task.models and MODEL in task.models


def test_model_only_mode_drops_stage1_from_prompt():
    task = load_task(TASK, destination=benchmark.LOCAL_DESTINATION, load_tables=[])
    cfg = ELTEnvConfig(task=task, model_name="Qwen/Qwen3-8B")
    assert cfg.stage1 is False
    prompt = load_prompt(cfg.destination, stage1=cfg.stage1)
    assert "already done" in prompt.instance

    task2 = load_task(TASK, destination=benchmark.LOCAL_DESTINATION, load_tables=None)
    cfg2 = ELTEnvConfig(task=task2, model_name="Qwen/Qwen3-8B")
    assert cfg2.stage1 is True


def test_execute_sql_session_context_targets_the_namespace():
    from eltbench.env import _session_context_sql

    ns, _block = plan_namespace(
        benchmark.LOCAL_DESTINATION, TASK, _local_base(), 0, RUN_TAG
    )
    pg = _session_context_sql(
        benchmark.LOCAL_DESTINATION, ns.eval_database, ns.eval_schema
    )
    assert pg == [f'SET search_path TO "{ns.eval_schema}"']

    sf_ns, _ = plan_namespace("snowflake", TASK, _sf_base(), 0, RUN_TAG)
    sf = _session_context_sql("snowflake", sf_ns.eval_database, sf_ns.physical_schema)
    assert sf == [
        f'USE DATABASE "{sf_ns.eval_database}"',
        'USE SCHEMA "AIRBYTE_SCHEMA"',
    ]

    dbx = _session_context_sql("databricks", "cat", "sch")
    assert dbx == ["USE CATALOG `cat`", "USE SCHEMA `sch`"]


def test_execute_sql_database_allowlist():
    from eltbench.env import _database_allowed_error, _readonly_context_sql

    ns, _ = plan_namespace(benchmark.LOCAL_DESTINATION, TASK, _local_base(), 0, RUN_TAG)
    assert _database_allowed_error(ns.eval_database, ns, TASK) is None
    assert _database_allowed_error(TASK, ns, TASK) is None
    assert _database_allowed_error("postgres", ns, TASK) is not None
    assert _database_allowed_error("other_task_db", ns, TASK) is not None

    assert _readonly_context_sql(benchmark.LOCAL_DESTINATION) == [
        "SET default_transaction_read_only = on"
    ]
    assert _readonly_context_sql("redshift") == [
        "SET default_transaction_read_only = on"
    ]

    assert _readonly_context_sql("snowflake") == []


def test_load_mode_resolution():
    from eltbench.env import ELTEnvConfig

    local = load_task(TASK, destination=benchmark.LOCAL_DESTINATION, load_tables=None)
    assert ELTEnvConfig(task=local, model_name="m").effective_load_mode == "content"
    assert (
        ELTEnvConfig(task=local, model_name="m", load_mode="count").effective_load_mode
        == "count"
    )

    sf = load_task(TASK, destination="snowflake", load_tables=None)
    assert (
        ELTEnvConfig(
            task=sf, model_name="m", destination="snowflake"
        ).effective_load_mode
        == "count"
    )

    with pytest.raises(ValueError):
        ELTEnvConfig(task=sf, model_name="m")

    plain = load_task(TASK, destination=benchmark.LOCAL_DESTINATION, load_tables=[])
    assert ELTEnvConfig(task=plain, model_name="m").stage1 is False


def test_container_implements_tinkers_sandbox_interface():
    from tinker_cookbook.sandbox import SandboxInterface, SandboxResult

    from eltbench.env import ELTContainer

    container = ELTContainer(Path("/tmp/eltbench-sandbox-probe"))
    assert isinstance(container, SandboxInterface)
    assert container.sandbox_id == container.name
    assert hasattr(container, "run_command") and hasattr(container, "cleanup")
    assert SandboxResult(stdout="x", stderr="", exit_code=0).exit_code == 0


def test_rollout_config_reuses_agentic_preset():
    from tinker_cookbook.rl.rollout_presets import agentic

    task = load_task(TASK, destination=benchmark.LOCAL_DESTINATION, load_tables=[])
    cfg = ELTEnvConfig(
        task=task,
        model_name="Qwen/Qwen3-8B",
        max_turns=7,
        max_trajectory_tokens=1234,
        max_tool_calls=5,
    )
    rc = build_rollout_config(cfg)
    assert (
        rc.limits.max_turns,
        rc.limits.max_trajectory_tokens,
        rc.limits.max_tool_calls,
    ) == (7, 1234, 5)
    base = agentic()
    assert rc.parse_errors == base.parse_errors
    assert rc.tool_execution == base.tool_execution == "sequential"

    assert rc.termination.zero_reward_on_limit is False
    assert base.termination.zero_reward_on_limit is True
    assert rc.termination.limit_stop_reasons == base.termination.limit_stop_reasons

    assert rollout_termination() == rc.termination


def test_builder_derives_run_dir_from_run_tag():
    task = load_task(TASK, destination=benchmark.LOCAL_DESTINATION, load_tables=[])
    cfg = ELTEnvConfig(
        task=task,
        model_name="Qwen/Qwen3-8B",
        run_tag="fixed1",
        group_size=3,
        runs_root=Path("/tmp/eltbench-runs"),
    )
    builder = ELTEnvGroupBuilder.__new__(ELTEnvGroupBuilder)
    builder.cfg = cfg
    builder.run_tag = cfg.run_tag
    builder.run_dir = Path(cfg.runs_root) / f"{cfg.task.name}-{builder.run_tag}"
    assert builder.run_dir.name == f"{TASK}-fixed1"
    names = {
        plan_namespace(
            cfg.destination, cfg.task.name, _local_base(), i, builder.run_tag
        )[0].eval_schema
        for i in range(cfg.group_size)
    }
    assert len(names) == cfg.group_size


def test_each_rollout_gets_its_own_tokenizer():
    from eltbench import env as env_module

    source = inspect.getsource(env_module.ELTEnvGroupBuilder.make_envs)
    assert "renderer=self.renderer" not in source, (
        "make_envs must not share a renderer instance"
    )
    assert "env_renderer" in source and "fresh_tokenizer(" in source

    ft = inspect.getsource(env_module.fresh_tokenizer)
    assert "AutoTokenizer.from_pretrained" in ft
    body = ft.split('"""')[-1]
    assert "get_tokenizer(" not in body, (
        "Use a fresh tokenizer instead of the cached instance"
    )


def test_group_builder_exposes_compute_group_rewards():
    from types import SimpleNamespace

    from eltbench.env import ELTEnvGroupBuilder

    assert hasattr(ELTEnvGroupBuilder, "compute_group_rewards")
    builder = object.__new__(ELTEnvGroupBuilder)
    traj = SimpleNamespace(
        transitions=[
            SimpleNamespace(reward=0.0, logs={"reward/parse_error": 1.0}),
            SimpleNamespace(reward=0.25, logs={"reward/model_partial": 0.25, "s": "x"}),
        ]
    )
    out = asyncio.run(builder.compute_group_rewards([traj], []))
    assert len(out) == 1
    reward, metrics = out[0]
    assert reward == pytest.approx(0.25)
    assert metrics == {"reward/model_partial": 0.25}


def test_tinker_adapts_swe_agent_file_tools(tmp_path, monkeypatch):
    """Tinker wrappers call the checked-in SWE-agent commands with isolated editor state."""
    import subprocess
    import uuid
    from types import SimpleNamespace

    import eltbench.env as env_module
    from eltbench.env import ELTTools

    tools_root = env_module.ROOT / "repo/agents/SWE-agent/tools"
    monkeypatch.setattr(env_module, "SWE_TOOLS_CONTAINER", str(tools_root))
    workspace = tmp_path / "workspace"
    (workspace / "elt/models").mkdir(parents=True)
    (workspace / "elt/main.tf").write_text(
        'terraform {\n  required_version = "1.0"\n}\n'
    )
    (workspace / "documentation").mkdir()
    (workspace / "documentation/setup.md").write_text("provider setup\n")
    registry_path = tmp_path / f"swe-registry-{uuid.uuid4().hex}.json"

    class FakeContainer:
        host_workspace = workspace
        name = "unit-test"

        @staticmethod
        def exec(command, timeout=600):
            command = command.replace("/workspace", str(workspace))
            command = command.replace("/tmp/elt-swe-unit-test.json", str(registry_path))
            result = subprocess.run(
                ["bash", "-lc", command],
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return result.returncode, result.stdout + result.stderr

    tools = ELTTools(
        FakeContainer(),
        load_task(TASK, destination=benchmark.LOCAL_DESTINATION, load_tables=[]),
        SimpleNamespace(namespace=None, harness_config={}, destination_config={}),
        benchmark.LOCAL_DESTINATION,
    )

    def run(fn, **kwargs) -> str:
        return str(asyncio.run(fn._fn(tools, **kwargs)))

    assert "terraform" in run(tools.open, path="elt/main.tf")
    assert "Text replaced" in run(tools.edit, search='"1.0"', replace='"1.1"')
    assert '"1.1"' in (workspace / "elt/main.tf").read_text()
    run(tools.insert, text="# inserted", line=1)
    assert "# inserted" in (workspace / "elt/main.tf").read_text()
    assert "No text supplied" in run(tools.insert, text="")
    assert "new.sql" in run(tools.create, filename="elt/models/new.sql")
    assert (workspace / "elt/models/new.sql").exists()
    assert "provider setup" in run(tools.open, path="documentation/setup.md")
    denied = run(tools.edit, search="provider setup", replace="bad")
    assert "restricted" in denied
    specs = {tool.name: tool.to_spec() for tool in tools.all()}
    assert {
        "open",
        "goto",
        "scroll_up",
        "scroll_down",
        "create",
        "edit",
        "insert",
        "find_file",
        "search_dir",
        "search_file",
    } <= specs.keys()
    assert "text" in specs["insert"]["parameters"]["required"]
    assert "file_text" not in specs["insert"]["parameters"]["properties"]
    assert "str_replace_editor" not in specs


def _hist(*calls):
    out = []
    for name, args in calls:
        fn = type("F", (), {"name": name, "arguments": json.dumps(args)})()
        out.append(
            {"role": "assistant", "tool_calls": [type("C", (), {"function": fn})()]}
        )
    return out


def test_process_signals_counts_bash_reads_of_docs():
    from eltbench.env import process_signals as ps

    view = ("open", {"path": "documentation/x.md"})
    searched_doc = (
        "search_file",
        {"search_term": "config", "file": "documentation/x.md"},
    )
    searched_doc_dir = (
        "search_dir",
        {"search_term": "config", "dir": "documentation/"},
    )
    cat_doc = (
        "bash",
        {"command": "cd /workspace && cat documentation/source_postgres.md"},
    )
    cat_cfg = ("bash", {"command": "head -40 config.yaml"})
    cat_schemas = ("bash", {"command": "cat schemas/state.csv"})
    write = ("create", {"filename": "elt/x.sql"})
    dbt = ("bash", {"command": "cd elt && dbt run"})

    assert ps(_hist(view, write)).score == 0.2
    assert ps(_hist(searched_doc, write)).score == 0.2
    assert ps(_hist(searched_doc_dir, write)).score == 0.2
    assert ps(_hist(cat_doc, write)).score == 0.2
    assert ps(_hist(cat_cfg, write)).score == 0.2
    assert ps(_hist(cat_schemas, write)).score == 0.2
    assert ps(_hist(write, cat_doc, dbt)).score == 0.1
    assert ps(_hist(write, dbt, cat_doc)).score == 0.0

    assert ps(_hist(("bash", {"command": "ls -la"}), write)).score == 0.0

    assert ps(_hist(("bash", {"command": "ls documentation/"}), write)).score == 0.0
    assert ps(_hist(("bash", {"command": "cat documentation/"}), write)).score == 0.0
    assert ps(_hist(("open", {"path": "documentation/"}), write)).score == 0.0


def _pg_config():
    return {
        "host": "localhost",
        "port": 5433,
        "user": "postgres",
        "password": "testelt",
        "database": "elt_warehouse",
        "schema": "public",
        "bootstrap_database": "postgres",
    }


@pytest.fixture(scope="module")
def pg():
    pytest.importorskip("psycopg2")
    from eltbench.destinations import PostgresConnector

    conn = PostgresConnector(_pg_config())
    try:
        conn.conn()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(
            f"Local PostgreSQL is unavailable: {type(exc).__name__}: {str(exc)[:80]}"
        )
    conn.close()
    return _pg_config()


def test_local_workspace_attaches_official_source_tables(pg, scratch, staging):
    import psycopg2

    manifest = benchmark.source_table_manifest(TASK)
    from eltbench.destinations import PostgresConnector

    src = PostgresConnector(pg)
    seeded = {}
    for name in sorted(manifest):
        try:
            size = src.table_size(TASK, "public", name)
        except Exception:  # noqa: BLE001
            continue
        if size == manifest[name]:
            seeded[name] = size
    src.close()
    if not seeded:
        pytest.skip(f"Source tables for {TASK} have not been seeded")

    from eltbench.train import local_destination_config, local_harness_config

    tag = f"t{RUN_TAG}"[:10]
    root = scratch / "ws-attach"
    ws = materialize_workspace(
        TASK,
        root,
        destination=benchmark.LOCAL_DESTINATION,
        index=0,
        run_tag=tag,
        staging_root=staging,
        base_destination_config=local_destination_config(),
        harness_config=local_harness_config(),
        prepare=True,
    )
    ns = ws.namespace
    try:
        conn = psycopg2.connect(
            host=pg["host"],
            port=pg["port"],
            user=pg["user"],
            password=pg["password"],
            dbname=ns.eval_database,
        )
        try:
            with conn.cursor() as cur:
                for name, want in seeded.items():
                    cur.execute(
                        'SELECT count(*) FROM "%s"."%s"' % (ns.eval_schema, name)
                    )
                    got = cur.fetchone()[0]
                    assert got == want, (
                        f"{ns.eval_schema}.{name} contains {got} rows (source has {want}); "
                        "the environment did not attach the source table to this rollout"
                    )
        finally:
            conn.close()
    finally:
        from eltbench.workspace import drop_namespace

        drop_namespace(benchmark.LOCAL_DESTINATION, ws.harness_config, ns)


def _bulk_insert(
    connector, database: str, schema: str, table: str, frame: pd.DataFrame
) -> None:
    from psycopg2.extras import execute_values

    connector.use_database(database)
    rows = [
        tuple("" if v is None else str(v) for v in row)
        for row in frame.itertuples(index=False, name=None)
    ]
    cur = connector.conn().cursor()
    try:
        execute_values(
            cur, f'INSERT INTO "{schema}"."{table}" VALUES %s', rows, page_size=1000
        )
    finally:
        cur.close()


def _load_gt_into_schema(connector, schema: str) -> None:
    gt = pd.read_csv(
        benchmark.GT_ROOT / TASK / f"{MODEL}.csv",
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    cols = ", ".join(f'"{c}" text' for c in gt.columns)
    connector.execute(f'DROP TABLE IF EXISTS "{schema}"."{MODEL}"')
    connector.execute(f'CREATE TABLE "{schema}"."{MODEL}" ({cols})')
    _bulk_insert(connector, connector.config["database"], schema, MODEL, gt)


def test_rewards_are_independent_per_rollout(pg, scratch):
    from eltbench.destinations import PostgresConnector

    ns_a, block_a = plan_namespace(
        benchmark.LOCAL_DESTINATION, TASK, _local_base(), 0, RUN_TAG
    )
    ns_b, block_b = plan_namespace(
        benchmark.LOCAL_DESTINATION, TASK, _local_base(), 1, RUN_TAG
    )
    assert (ns_a.eval_database, ns_a.eval_schema) != (
        ns_b.eval_database,
        ns_b.eval_schema,
    )

    connector = PostgresConnector(pg)
    try:
        for block in (block_a, block_b):
            connector.reset_namespace(block["database"], block["schema"])
        _load_gt_into_schema(connector, ns_a.eval_schema)

        task = load_task(TASK, destination=benchmark.LOCAL_DESTINATION, load_tables=[])
        reward_a = compute_reward(
            connector, task.name, task.gt_dir, ns_a.eval_database, ns_a.eval_schema
        )
        reward_b = compute_reward(
            connector, task.name, task.gt_dir, ns_b.eval_database, ns_b.eval_schema
        )
        assert reward_a.model_score == 1.0, reward_a.summary()
        assert reward_b.model_score == 0.0, reward_b.summary()
        assert reward_a.total != reward_b.total

        connector.reset_namespace(block_b["database"], block_b["schema"])
        reward_a2 = compute_reward(
            connector, task.name, task.gt_dir, ns_a.eval_database, ns_a.eval_schema
        )
        assert reward_a2.model_score == 1.0
    finally:
        for block in (block_a, block_b):
            connector.drop_namespace(block["database"], block["schema"])
        connector.close()


def test_stage1_load_check_both_modes(pg, scratch):
    from eltbench.destinations import PostgresConnector
    from eltbench.reward import source_snapshot

    connector = PostgresConnector(pg)
    manifest = benchmark.source_table_manifest(TASK)

    table = None
    for name, expected in sorted(manifest.items()):
        try:
            if connector.table_size(TASK, "public", name) == expected:
                table = name
                break
        except Exception:  # noqa: BLE001
            continue
    if table is None:
        connector.close()
        pytest.skip(
            f"Source tables for {TASK} do not match table.json (are they seeded?)"
        )

    ns, block = plan_namespace(
        benchmark.LOCAL_DESTINATION, TASK, _local_base(), 2, RUN_TAG
    )
    try:
        want = connector.fetch_table(TASK, "public", table)
        connector.reset_namespace(block["database"], block["schema"])
        cols = ", ".join(f'"{c}" text' for c in want.columns)
        connector.execute(f'CREATE TABLE "{ns.eval_schema}"."{table}" ({cols})')
        _bulk_insert(connector, ns.eval_database, ns.eval_schema, table, want)

        task = load_task(
            TASK, destination=benchmark.LOCAL_DESTINATION, load_tables=None
        )
        n_tables = len(task.load_tables)

        snapshot = source_snapshot(
            connector, task.source_db, task.source_schema, list(task.load_tables)
        )

        def grade(mode: str):
            return compute_reward(
                connector,
                task.name,
                task.gt_dir,
                ns.eval_database,
                ns.eval_schema,
                source_db=task.source_db,
                source_schema="public",
                load_tables=task.load_tables,
                load_mode=mode,
                source_hashes=snapshot,
            )

        reward = grade("content")
        assert reward.load_evaluated and reward.load_mode == "content"
        assert reward.load_count_score == pytest.approx(1 / n_tables)
        assert reward.load_content_score == pytest.approx(1 / n_tables)
        assert reward.load_score == pytest.approx(1 / n_tables)
        assert table in [t.table for t in reward.tables if t.ok]

        first_col = str(want.columns[0])
        connector.use_database(ns.eval_database)
        connector.execute(
            f'UPDATE "{ns.eval_schema}"."{table}" '
            f"SET \"{first_col}\" = '__CORRUPTED__' "
            f'WHERE ctid IN (SELECT ctid FROM "{ns.eval_schema}"."{table}" LIMIT 1)'
        )

        by_content = grade("content")
        assert by_content.load_content_score == 0.0
        assert by_content.load_count_score == pytest.approx(1 / n_tables)

        by_count = grade("count")
        assert by_count.load_mode == "count"
        assert by_count.load_score == pytest.approx(1 / n_tables)
        assert by_count.load_content_score is None
        assert table in [t.table for t in by_count.tables if t.ok]

        assert manifest[table] == connector.table_size(TASK, "public", table)
    finally:
        connector.drop_namespace(block["database"], block["schema"])
        connector.close()
