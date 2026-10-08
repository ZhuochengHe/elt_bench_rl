"""A local rollout must not reach the model with an incomplete source namespace."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eltbench import benchmark, localsources  # noqa: E402
from eltbench.workspace import (  # noqa: E402
    RolloutNamespace,
    _validate_local_source_tables,
)


class _Cursor:
    def __init__(self, counts):
        self.counts = iter(counts)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, *_args):
        self.current = next(self.counts)

    def fetchone(self):
        return (self.current,)


class _Connection:
    def __init__(self, counts):
        self.counts = counts

    def cursor(self):
        return _Cursor(self.counts)

    def close(self):
        pass


def test_local_source_validation_rejects_wrong_manifest_row_count(monkeypatch):
    monkeypatch.setattr(
        benchmark, "source_table_manifest", lambda _task: {"present": 3, "short": 4}
    )
    monkeypatch.setattr("psycopg2.connect", lambda **_kwargs: _Connection([3, 2]))
    ns = RolloutNamespace("local_postgres", "task", 0, "warehouse", "rollout")

    with pytest.raises(RuntimeError, match=r"short: got 2 rows, expected 4"):
        _validate_local_source_tables(
            "task",
            ns,
            {"host": "localhost", "port": 5432, "user": "u", "password": "p"},
        )


def test_non_postgres_source_fetch_failure_is_not_silently_skipped(monkeypatch):
    monkeypatch.setattr("psycopg2.connect", lambda **_kwargs: _Connection([]))
    monkeypatch.setattr(localsources, "_rows_from_raw_seed", lambda *_args: None)

    def fail_fetch(*_args):
        raise OSError("source unavailable")

    monkeypatch.setattr(localsources, "_rows_from_custom_api", fail_fetch)
    config = {"custom_api": {"config": {"tables": ["source_table"]}}}

    with pytest.raises(RuntimeError, match="source_table.*source unavailable"):
        localsources.materialize_non_postgres_sources(
            config,
            target_db="warehouse",
            target_schema="rollout",
            task="task",
            harness={"host": "localhost", "port": 5432, "user": "u", "password": "p"},
        )
