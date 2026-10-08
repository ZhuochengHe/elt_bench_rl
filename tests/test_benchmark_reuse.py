"""Test official benchmark adapters and comparison semantics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eltbench import benchmark  # noqa: E402
from eltbench.prompt import load_prompt  # noqa: E402
from eltbench.reward import model_outcome  # noqa: E402

TASK = "address"
MODEL = "states"


def test_official_modules_are_imported():
    assert benchmark.db_connectors.__name__ == "db_connectors"
    assert benchmark.eva_stage2.__name__ == "eva_stage2"
    assert benchmark.write_config.__name__ == "write_config"

    assert Path(benchmark.db_connectors.__file__).is_relative_to(benchmark.REPO)
    assert Path(benchmark.eva_stage2.__file__).is_relative_to(benchmark.REPO)
    assert Path(benchmark.write_config.__file__).is_relative_to(benchmark.REPO)


def test_local_destination_registered_in_official_registry():
    assert benchmark.LOCAL_DESTINATION in benchmark.DESTINATIONS
    spec = benchmark.get_destination(benchmark.LOCAL_DESTINATION)
    assert spec.display_name == "PostgreSQL"

    adapted = benchmark.official_adapt_prompt(
        "loaded into Snowflake.", "local_postgres"
    )
    assert "PostgreSQL" in adapted and "Snowflake" not in adapted


def test_prepare_destination_delegates_to_official(monkeypatch):
    calls = {}

    def fake(destination, task_input_dir, credential_path=None):
        calls["args"] = (destination, Path(task_input_dir), credential_path)
        return "DB.SCHEMA"

    monkeypatch.setattr(benchmark, "_official_prepare_destination", fake)
    out = benchmark.prepare_destination(
        "snowflake", "/tmp/nonexistent", "/tmp/cred.json"
    )
    assert out == "DB.SCHEMA"
    assert calls["args"] == ("snowflake", Path("/tmp/nonexistent"), "/tmp/cred.json")


def test_official_private_symbol_is_pinned():
    from common import warehouse

    assert callable(getattr(warehouse, "_load_destination_config", None))


def test_table_manifest_comes_from_official_table_json():
    official = json.loads(benchmark.TABLE_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert benchmark.source_table_manifest(TASK) == official[TASK] or set(
        benchmark.source_tables(TASK)
    ) == set(official[TASK])
    assert "state" in benchmark.source_tables(TASK)


def test_sort_keys_come_from_official_loader():
    official = benchmark.eva_stage2.load_sort_keys(str(benchmark.SORT_KEY_PATH))
    assert benchmark.task_sort_keys(TASK) == official[TASK]
    assert benchmark.model_sort_keys(TASK, MODEL) == official[TASK][MODEL]


def test_prompt_is_rendered_and_destination_adapted():
    prompt = load_prompt(benchmark.LOCAL_DESTINATION, stage1=False)
    assert "{{" not in prompt.system and "{{" not in prompt.instance
    assert "PostgreSQL" in prompt.system

    assert "bash-$ is only how a human would" in prompt.system
    assert "Your shell prompt is formatted as follows" not in prompt.system

    assert "Stage 1: Extraction and Loading (already done)" in prompt.instance
    assert "Initialize the Airbyte Provider" not in prompt.instance

    cloud = load_prompt("snowflake", stage1=True)
    assert "Initialize the Airbyte Provider" in cloud.instance


class _FakeConnector:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame

    def fetch_table(self, database, schema, table):  # noqa: ARG002
        return self.frame.copy()


def _outcome(gt: pd.DataFrame, got: pd.DataFrame, tmp_path: Path):
    gt_path = tmp_path / f"{MODEL}.csv"
    gt.to_csv(gt_path, index=False)
    return model_outcome(
        _FakeConnector(got),
        "db",
        "schema",
        MODEL,
        gt_path,
        benchmark.model_sort_keys(TASK, MODEL),
    )


def _official_match(gt: pd.DataFrame, got: pd.DataFrame) -> bool:
    return bool(benchmark.official_check_correctness(gt, got)["match"])


BASE = pd.DataFrame(
    {
        "abbreviation": ["AK", "AL", "CA"],
        "num_asian": ["100", "200", "300"],
        "ratio": ["1.50", "2.25", "3.75"],
    }
)


def test_identical_frames_match_in_both(tmp_path):
    assert _outcome(BASE, BASE, tmp_path).ok
    assert _official_match(BASE, BASE)


def test_format_only_differences_match_in_both(tmp_path):
    got = BASE.copy()
    got["num_asian"] = ["1e2", "2.0e2", "300.0"]
    assert _outcome(BASE, got, tmp_path).ok
    assert _official_match(BASE, got)


def test_we_are_stricter_than_official_on_scaled_numbers(tmp_path):
    got = BASE.copy()
    got["num_asian"] = [f"{int(v) * 1.005:.6f}" for v in BASE["num_asian"]]
    assert _official_match(BASE, got) is True
    assert _outcome(BASE, got, tmp_path).ok is False


def test_we_accept_null_literals_official_rejects(tmp_path):
    gt = pd.DataFrame({"a": ["x", ""], "b": ["1", "2"]})
    got = pd.DataFrame({"a": ["x", "NULL"], "b": ["1", "2"]})
    assert _official_match(gt, got) is False
    assert _outcome(gt, got, tmp_path).ok is True


def test_real_differences_are_caught_by_both(tmp_path):
    got = BASE.copy()
    got.loc[1, "ratio"] = "99.99"
    assert _official_match(BASE, got) is False
    assert _outcome(BASE, got, tmp_path).ok is False


def test_missing_column_is_caught(tmp_path):
    got = BASE.drop(columns=["ratio"])
    assert _official_match(BASE, got) is False
    out = _outcome(BASE, got, tmp_path)
    assert not out.ok and out.missing == ["ratio"]


def test_row_count_mismatch_is_caught(tmp_path):
    got = BASE.iloc[:2].copy()
    assert _official_match(BASE, got) is False
    assert not _outcome(BASE, got, tmp_path).ok


def test_long_identifier_is_matched_after_postgres_truncation(tmp_path):
    long_col = "c" * 70
    gt = pd.DataFrame({long_col: ["1", "2"]})
    got = pd.DataFrame({long_col[:63]: ["1", "2"]})
    assert _outcome(gt, got, tmp_path).ok


def test_load_mode_is_content_only_where_a_separate_source_exists():
    assert benchmark.load_mode_for(benchmark.LOCAL_DESTINATION) == "content"
    for destination in ("snowflake", "databricks", "redshift"):
        assert benchmark.load_mode_for(destination) == "count"


@pytest.mark.parametrize("destination", ["snowflake", "databricks", "redshift"])
def test_official_backends_reuse_credential_file(destination):
    path = benchmark.credential_path_for(destination)
    assert path.name == f"{destination}_credential.json"
    cfg = benchmark.connector_config(destination)
    if path.is_file():
        assert cfg == json.loads(path.read_text(encoding="utf-8"))


def test_credential_directory_can_live_outside_the_submodule(monkeypatch, tmp_path):
    monkeypatch.setenv("ELT_BENCH_CREDENTIAL_DIR", str(tmp_path))
    assert benchmark.credential_path_for("snowflake") == (
        tmp_path / "snowflake_credential.json"
    )


def test_staged_inputs_use_external_credential_directory(monkeypatch, tmp_path):
    from eltbench.workspace import stage_inputs

    credential_dir = tmp_path / "secrets"
    monkeypatch.setenv("ELT_BENCH_CREDENTIAL_DIR", str(credential_dir))
    calls = []

    def generate(destination, *, dest, credential_path):
        calls.append((destination, credential_path))
        dest.mkdir(parents=True, exist_ok=True)
        return 1, dest

    monkeypatch.setattr(benchmark, "generate_inputs", generate)
    stage_inputs("snowflake", tmp_path / "staged")

    assert calls == [("snowflake", credential_dir / "snowflake_credential.json")]
