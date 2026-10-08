"""Run a credential-free local integration rollout through the real environment.

The scripted policy exercises workspace generation, the container tool interface,
dbt materialization, warehouse reads, and reward computation without model sampling
or Tinker billing. The gold and typo cases use a test fixture as dbt input; the
separate rollout_local_sources.py check validates transformations from materialized
source tables.

Usage: python tests/manual/rollout_local.py [gold|typo|empty|all]

Prerequisites: local benchmark services, the elt-swe:local image, and seeded source
data. Missing prerequisites are reported as SKIP.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eltbench import benchmark  # noqa: E402
from eltbench.env import ELTEnvConfig, ELTEnvGroupBuilder, load_task  # noqa: E402
from eltbench.train import local_destination_config, local_harness_config  # noqa: E402
from eltbench.workspace import new_run_tag  # noqa: E402

TASK = "address"
MODEL = "states"
MODEL_NAME = "Qwen/Qwen3-8B"
IMAGE = "elt-swe:local"


def preconditions() -> list[str]:
    import subprocess

    missing = []
    if (
        subprocess.run(
            ["docker", "image", "inspect", IMAGE], capture_output=True
        ).returncode
        != 0
    ):
        missing.append(
            f"Image {IMAGE} is missing; build docker/Dockerfile.elt-swe first."
        )
    if (
        subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", "elt-postgres"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        != "true"
    ):
        missing.append(
            "Container elt-postgres is not running; start the benchmark Compose services."
        )
    try:
        import psycopg2

        psycopg2.connect(
            host=local_harness_config()["host"],
            port=local_harness_config()["port"],
            user="postgres",
            password="testelt",
            dbname="postgres",
            connect_timeout=5,
        ).close()
    except Exception as exc:  # noqa: BLE001
        missing.append(
            f"Cannot connect to local PostgreSQL ({type(exc).__name__}: {exc})"
        )
    return missing


def _assistant_call(name: str, args: dict, idx: int):
    from tinker_cookbook.renderers.base import ToolCall

    return {
        "role": "assistant",
        "content": f"(call {name})",
        "tool_calls": [
            ToolCall(
                function=ToolCall.FunctionBody(
                    name=name, arguments=json.dumps(args, ensure_ascii=False)
                ),
                id=f"call_{idx}",
            )
        ],
    }


def _sql_select(
    schema: str, source_table: str, columns: list[str], rename_first: str | None = None
) -> str:
    selects = []
    for i, c in enumerate(columns):
        alias = rename_first if (i == 0 and rename_first) else c
        selects.append(f'"{c}" as "{alias}"')
    body = ",\n    ".join(selects)
    return f'select\n    {body}\nfrom "{schema}"."{source_table}"\n'


def script_for(
    kind: str, destination: dict, schema: str, fixture: str, gt_columns: list[str]
) -> list:
    if kind == "empty":
        return [_assistant_call("submit", {}, 0)]

    project = """name: eltbench_probe
version: "1.0"
profile: eltbench_probe
model-paths: ["models"]
models:
  eltbench_probe:
    +materialized: table
"""

    profile = (
        "eltbench_probe:\n"
        "  target: dev\n"
        "  outputs:\n"
        "    dev:\n"
        "      type: postgres\n"
        f"      host: {destination['host']}\n"
        f"      port: {destination['port']}\n"
        f"      user: {destination['user']}\n"
        f"      password: {destination['password']}\n"
        f"      dbname: {destination['database']}\n"
        f"      schema: {destination['schema']}\n"
        "      threads: 1\n"
    )
    sql = _sql_select(
        schema, fixture, gt_columns, rename_first="typo_col" if kind == "typo" else None
    )
    return [
        _assistant_call("create", {"filename": "elt/dbt_project.yml"}, 0),
        _assistant_call("insert", {"text": project}, 1),
        _assistant_call("create", {"filename": "elt/profiles.yml"}, 2),
        _assistant_call("insert", {"text": profile}, 3),
        _assistant_call("bash", {"command": "mkdir -p /workspace/elt/models"}, 4),
        _assistant_call("create", {"filename": f"elt/models/{MODEL}.sql"}, 5),
        _assistant_call("insert", {"text": sql}, 6),
        _assistant_call(
            "bash",
            {
                "command": "cd /workspace/elt && dbt run --profiles-dir . "
                "2>&1 | tail -20"
            },
            7,
        ),
        _assistant_call("submit", {}, 8),
    ]


def _tool_name(action: dict) -> str:
    calls = action.get("tool_calls") or []
    if not calls:
        return ""
    call = calls[0]
    fn = getattr(call, "function", None) or (
        call.get("function") if isinstance(call, dict) else None
    )
    return str(getattr(fn, "name", "") or (fn or {}).get("name", ""))


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(part.get("text", part)) if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


async def run_case(kind: str) -> tuple[float, str]:
    import psycopg2

    run_tag = new_run_tag()
    task = load_task(TASK, destination=benchmark.LOCAL_DESTINATION, load_tables=[])
    cfg = ELTEnvConfig(
        task=task,
        model_name=MODEL_NAME,
        destination=benchmark.LOCAL_DESTINATION,
        group_size=1,
        max_turns=12,
        base_destination_config=local_destination_config(),
        harness_destination_config=local_harness_config(),
        run_tag=run_tag,
    )
    builder = ELTEnvGroupBuilder(cfg, _tokenizer())
    envs = await builder.make_envs()
    ws = builder.rollouts[0][0]
    ns = ws.namespace
    print(f"    Rollout namespace: {ns.eval_database}.{ns.eval_schema}")
    print(f"    Workspace: {ws.root}")

    gt = pd.read_csv(
        task.gt_dir / f"{MODEL}.csv",
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    fixture = f"{MODEL}__fixture"
    host = local_harness_config()
    conn = psycopg2.connect(
        host=host["host"],
        port=host["port"],
        user="postgres",
        password="testelt",
        dbname=ns.eval_database,
    )
    conn.autocommit = True
    cur = conn.cursor()
    cols = ", ".join(f'"{c}" text' for c in gt.columns)
    cur.execute(f'CREATE TABLE "{ns.eval_schema}"."{fixture}" ({cols})')
    cur.executemany(
        f'INSERT INTO "{ns.eval_schema}"."{fixture}" VALUES '
        f"({', '.join(['%s'] * len(gt.columns))})",
        [tuple(r) for r in gt.itertuples(index=False, name=None)],
    )
    cur.close()
    conn.close()

    config = yaml.safe_load((ws.root / "config.yaml").read_text(encoding="utf-8"))
    destination = config[benchmark.LOCAL_DESTINATION]["config"]
    assert destination["schema"] == ns.eval_schema, (
        "Workspace does not target this rollout namespace"
    )
    actions = script_for(kind, destination, ns.eval_schema, fixture, list(gt.columns))

    inner = envs[0].message_env
    try:
        await inner.initial_observation()
        reward, metrics, last_bash = None, {}, ""
        consumed = 0
        for i, action in enumerate(actions):
            name = _tool_name(action)
            res = await inner.step(action)
            for m in res.next_messages[consumed:]:
                if m.get("role") == "tool":
                    txt = _content_text(m.get("content"))
                    if name == "bash":
                        last_bash = txt
                    summary = (
                        "file operation completed"
                        if name in {"create", "insert"}
                        else txt
                    )
                    print(
                        f"    step{i} tool result: {summary[:400].replace(chr(10), ' | ')}"
                    )
            consumed = len(res.next_messages)
            print(f"    step{i}: done={res.episode_done} reward={res.reward:.3f}")
            if res.episode_done:
                reward, metrics = res.reward, dict(res.metrics or {})
                break
    finally:
        await builder.cleanup()

    srdt = metrics.get("reward/model_srdt")
    partial = metrics.get("reward/model_partial")
    dbt_stage = metrics.get("trajectory/stage_dbt_materialized")
    return {
        "total": reward if reward is not None else 0.0,
        "srdt": float(srdt or 0.0),
        "partial": float(partial or 0.0),
        "dbt_stage": float(dbt_stage or 0.0),
        "detail": (
            f"rollout={ns.eval_schema} srdt={srdt} partial={partial} "
            f"dbt_stage={dbt_stage} "
            f"bash_tail={last_bash[-160:]!r}"
        ),
    }


def _tokenizer():
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    return get_tokenizer(MODEL_NAME)


async def main() -> int:
    missing = preconditions()
    if missing:
        print("SKIP: prerequisites are not met:")
        for m in missing:
            print(f"  - {m}")
        return 0

    kinds = sys.argv[1:] or ["all"]
    if kinds == ["all"]:
        kinds = ["gold", "typo", "empty"]

    gt = pd.read_csv(
        ROOT / "repo" / "evaluation" / "gt" / TASK / f"{MODEL}.csv",
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    n_cols = len(gt.columns)

    expect_srdt = {"gold": 1.0, "typo": 0.0, "empty": 0.0}

    expect_total = {"gold": 1.05, "typo": (n_cols - 1) / n_cols + 0.05, "empty": 0.0}

    bad = []
    for k in kinds:
        print(f"\n=== {k} ===")
        try:
            got = await run_case(k)
        except Exception as exc:  # noqa: BLE001
            print(f"    ERROR: {type(exc).__name__}: {exc}")
            bad.append(k)
            continue
        ok_srdt = abs(got["srdt"] - expect_srdt[k]) < 1e-6
        ok_total = abs(got["total"] - expect_total[k]) < 1e-6
        ok_dbt = abs(got["dbt_stage"] - (1.0 if k != "empty" else 0.0)) < 1e-6
        ok = ok_srdt and ok_total and ok_dbt
        print(
            f"    srdt={got['srdt']:.4f} (expected {expect_srdt[k]:.4f}) "
            f"total={got['total']:.4f} (expected {expect_total[k]:.4f})  "
            f"{'✅' if ok else '❌'} {got['detail']}"
        )
        if not ok:
            bad.append(k)
    print()
    print("All cases match expectations." if not bad else f"Unexpected results: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
