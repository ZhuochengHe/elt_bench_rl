#!/usr/bin/env python3
"""Verify source-derived state columns through dbt and the local grader.

Unlike rollout_local.py, this check does not inject ground-truth fixtures. It
computes three verified output columns from source tables. Four other columns
are intentionally omitted because their exact benchmark semantics are unresolved.

Usage: python tests/manual/rollout_local_sources.py
"""

from __future__ import annotations

import asyncio
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rollout_local import (  # noqa: E402
    _assistant_call,
    _content_text,  # noqa: E402
    _tokenizer,
    _tool_name,
)

from eltbench import benchmark  # noqa: E402
from eltbench.env import ELTEnvConfig, ELTEnvGroupBuilder, load_task  # noqa: E402
from eltbench.reward import model_outcome  # noqa: E402
from eltbench.train import local_destination_config, local_harness_config  # noqa: E402
from eltbench.workspace import new_run_tag  # noqa: E402

TASK = "address"
MODEL = "states"
MODEL_NAME = "Qwen/Qwen3-8B"
EXPECTED_COLUMNS = 3
TOTAL_COLUMNS = 7
EXPECTED_PARTIAL = EXPECTED_COLUMNS / TOTAL_COLUMNS

MODEL_SQL = """\
with base as (
    select abbreviation, name from {{ this.schema }}.state
),
cty as (
    select upper(state) as key, count(distinct county) as num_counties
    from {{ this.schema }}.country
    group by 1
)
select b.abbreviation                                   as ABBREVIATION,
       b.name                                           as NAME,
       coalesce(c.num_counties, 0)                      as NUM_COUNTIES
from base b
left join cty c on c.key in (upper(b.abbreviation), upper(b.name))
"""

PROJECT = """name: eltbench_probe
version: "1.0"
profile: eltbench_probe
model-paths: ["models"]
models:
  eltbench_probe:
    +materialized: table
"""


def _profile(block: dict) -> str:
    return (
        "eltbench_probe:\n  target: dev\n  outputs:\n    dev:\n"
        "      type: postgres\n"
        f"      host: {block['host']}\n      port: {block['port']}\n"
        f"      user: {block['user']}\n      password: {block['password']}\n"
        f"      dbname: {block['database']}\n      schema: {block['schema']}\n"
        "      threads: 1\n"
    )


async def main() -> int:
    run_tag = new_run_tag()
    root = ROOT / "runs" / f"gold-src-{run_tag}"
    shutil.rmtree(root, ignore_errors=True)
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
    try:
        block = ws.destination_config
        actions = [
            _assistant_call("create", {"filename": "elt/dbt_project.yml"}, 0),
            _assistant_call("insert", {"text": PROJECT}, 1),
            _assistant_call("create", {"filename": "elt/profiles.yml"}, 2),
            _assistant_call("insert", {"text": _profile(block)}, 3),
            _assistant_call("bash", {"command": "mkdir -p /workspace/elt/models"}, 4),
            _assistant_call("create", {"filename": f"elt/models/{MODEL}.sql"}, 5),
            _assistant_call(
                "insert",
                {
                    "text": MODEL_SQL.replace(
                        "{{ this.schema }}", f'"{block["schema"]}"'
                    )
                },
                6,
            ),
            _assistant_call(
                "bash",
                {
                    "command": "cd /workspace/elt && dbt run --profiles-dir . "
                    "2>&1 | tail -6"
                },
                7,
            ),
            _assistant_call("submit", {}, 8),
        ]
        inner = envs[0].message_env
        await inner.initial_observation()
        reward, metrics = None, {}
        consumed = 0
        for i, action in enumerate(actions):
            name = _tool_name(action)
            res = await inner.step(action)
            for m in res.next_messages[consumed:]:
                if m.get("role") == "tool":
                    summary = (
                        "file operation completed"
                        if name in {"create", "insert"}
                        else _content_text(m.get("content"))
                    )
                    print(f"  step{i} {name}: {summary[:200].replace(chr(10), ' | ')}")
            consumed = len(res.next_messages)
            if res.episode_done:
                reward, metrics = res.reward, dict(res.metrics or {})
                break
        reader = builder.rollouts[0][2]
        outcome = model_outcome(
            reader,
            ws.namespace.eval_database,
            ws.namespace.eval_schema,
            MODEL,
            task.gt_dir / f"{MODEL}.csv",
            benchmark.task_sort_keys(TASK).get(MODEL, []),
        )
        print(
            f"  Matched columns {outcome.columns_matched}/{outcome.columns_total}; "
            f"wrong={outcome.wrong}; missing={outcome.missing}; "
            f"rows={outcome.got_rows}/{outcome.gt_rows}"
        )
    finally:
        await builder.cleanup()
        shutil.rmtree(root, ignore_errors=True)

    partial = float(metrics.get("reward/model_partial") or 0.0)
    srdt = float(metrics.get("reward/model_srdt") or 0.0)
    print(
        f"\n  model_srdt={srdt:.4f}  model_partial={partial:.4f}  "
        f"episode_total={reward:.4f}"
    )
    print(
        f"  Expected partial = {EXPECTED_COLUMNS}/{TOTAL_COLUMNS} = "
        f"{EXPECTED_PARTIAL:.4f} (three verified columns; four intentionally omitted)"
    )
    ok = abs(partial - EXPECTED_PARTIAL) < 1e-6
    print(
        "  Source-to-dbt-to-warehouse grading passed with the predicted score."
        if ok
        else "  Score differs from expectation; inspect the dbt output above."
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
