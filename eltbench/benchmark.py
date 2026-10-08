"Adapters for the pinned ELT-Bench implementation."

from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "repo"

_AGENTS_DIR = REPO / "agents"
_SWE_DIR = _AGENTS_DIR / "SWE-agent"
_EVAL_DIR = REPO / "evaluation"
_SETUP_DIR = REPO / "setup"

for _path in (_SWE_DIR, _AGENTS_DIR, _EVAL_DIR, _SETUP_DIR):
    if _path.is_dir() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

if not (_AGENTS_DIR / "common" / "warehouse.py").is_file():
    raise RuntimeError(
        f"ELT-Bench source file is missing: {_AGENTS_DIR / 'common' / 'warehouse.py'}.\n"
        "Initialize the pinned submodule at repo/:\n"
        "    git clone https://github.com/uiuc-kang-lab/ELT-Bench.git repo\n"
        "    git -C repo checkout fcf3129\n"
        "Task specs, ground truth, and credential templates are included; download setup archives separately."
    )


import db_connectors  # noqa: E402
import eva_stage2  # noqa: E402
import write_config  # noqa: E402
from common import (  # noqa: E402
    DESTINATIONS,
    DestinationSpec,
    adapt_prompt,
    destination_choices,
    get_destination,
    resolve_benchmark_path,
)
from common import prepare_destination as _official_prepare_destination  # noqa: E402
from common import warehouse as _warehouse  # noqa: E402

TABLE_MANIFEST_PATH = _EVAL_DIR / "table.json"
SORT_KEY_PATH = _EVAL_DIR / "sort_key.json"
GT_ROOT = ROOT / "runs" / "benchmark-assets" / "evaluation" / "gt"
BENCHMARK_ROOT = REPO / "elt-bench"
DOCUMENTATION_DIR = REPO / "documentation"
CREDENTIAL_DIR = _SETUP_DIR / "destination"


LOCAL_DESTINATION = "local_postgres"


LOCAL_FIXTURE_DESTINATION = "snowflake"


def fixture_destination(destination: str) -> str:
    return (
        LOCAL_FIXTURE_DESTINATION if destination == LOCAL_DESTINATION else destination
    )


def register_local_destination() -> None:
    DESTINATIONS.setdefault(
        LOCAL_DESTINATION,
        DestinationSpec(
            name=LOCAL_DESTINATION,
            display_name="PostgreSQL",
            inputs_dir="inputs_postgres",
            benchmark_dir=f"elt-bench/{LOCAL_FIXTURE_DESTINATION}",
            sync_delay_seconds=0,
        ),
    )


register_local_destination()


def read_destination_config(
    destination: str,
    task_input_dir: str | Path,
    credential_path: str | Path | None = None,
) -> dict[str, Any]:
    return _warehouse._load_destination_config(
        destination, Path(task_input_dir), credential_path
    )


def prepare_destination(
    destination: str,
    task_input_dir: str | Path,
    credential_path: str | Path | None = None,
    config_override: dict[str, Any] | None = None,
) -> str:
    if destination == LOCAL_DESTINATION:
        from .destinations import reset_postgres_namespace

        config = (
            dict(config_override)
            if config_override
            else read_destination_config(destination, task_input_dir, credential_path)
        )
        return reset_postgres_namespace(config)
    return _official_prepare_destination(destination, task_input_dir, credential_path)


def credential_path_for(destination: str) -> Path:
    credential_dir = Path(os.environ.get("ELT_BENCH_CREDENTIAL_DIR", CREDENTIAL_DIR))
    return credential_dir / f"{fixture_destination(destination)}_credential.json"


def connector_config(
    destination: str,
    *,
    workspace_config: dict[str, Any] | None = None,
    credential_path: str | Path | None = None,
) -> dict[str, Any]:
    if destination == LOCAL_DESTINATION:
        if workspace_config is None:
            raise ValueError(
                "The local backend requires workspace_config from config.yaml."
            )
        return dict(workspace_config)
    path = (
        Path(credential_path) if credential_path else credential_path_for(destination)
    )
    return json.loads(Path(path).read_text(encoding="utf-8"))


def generate_inputs(
    destination: str,
    dest: str | Path,
    credential_path: str | Path | None = None,
) -> tuple[int, Path]:
    return write_config.generate_inputs(
        fixture_destination(destination),
        dest=Path(dest),
        credential_path=credential_path,
    )


def benchmark_dir(destination: str) -> Path:
    return resolve_benchmark_path(fixture_destination(destination))


def load_mode_for(destination: str) -> str:
    return "content" if destination == LOCAL_DESTINATION else "count"


@lru_cache(maxsize=1)
def _table_manifest() -> dict[str, dict[str, int]]:
    return json.loads(TABLE_MANIFEST_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _sort_keys() -> dict[str, dict[str, list[str]]]:
    return eva_stage2.load_sort_keys(str(SORT_KEY_PATH))


def source_table_manifest(task: str) -> dict[str, int]:
    return dict(_table_manifest().get(task, {}))


def source_tables(task: str) -> list[str]:
    return sorted(source_table_manifest(task))


def task_sort_keys(task: str) -> dict[str, list[str]]:
    return dict(_sort_keys().get(task, {}))


def task_models(task: str) -> list[str]:
    return sorted(task_sort_keys(task))


def model_sort_keys(task: str, model: str) -> list[str]:
    return list(task_sort_keys(task).get(model, []))


def official_check_correctness(gt_df, got_df) -> dict:
    return eva_stage2.check_corretness(gt_df, got_df)


def official_sort_by_keys(df, keys: list[str]):
    return eva_stage2.sort_by_keys(df, keys)


def official_adapt_prompt(text: str, destination: str) -> str:
    return adapt_prompt(text, destination)


__all__ = [
    "BENCHMARK_ROOT",
    "CREDENTIAL_DIR",
    "DOCUMENTATION_DIR",
    "DESTINATIONS",
    "GT_ROOT",
    "LOCAL_DESTINATION",
    "LOCAL_FIXTURE_DESTINATION",
    "REPO",
    "ROOT",
    "SORT_KEY_PATH",
    "TABLE_MANIFEST_PATH",
    "adapt_prompt",
    "benchmark_dir",
    "connector_config",
    "credential_path_for",
    "db_connectors",
    "destination_choices",
    "eva_stage2",
    "fixture_destination",
    "generate_inputs",
    "get_destination",
    "model_sort_keys",
    "official_adapt_prompt",
    "official_check_correctness",
    "official_sort_by_keys",
    "prepare_destination",
    "read_destination_config",
    "resolve_benchmark_path",
    "source_table_manifest",
    "source_tables",
    "task_models",
    "task_sort_keys",
    "write_config",
]
