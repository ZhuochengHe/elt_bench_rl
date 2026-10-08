"""Test warehouse reader behavior across supported destinations."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "repo" / "evaluation"))

from eltbench.destinations import PostgresConnector, TableReader  # noqa: E402

SF_CRED = ROOT / "repo" / "setup" / "destination" / "snowflake_credential.json"
SF_PROBE_DB = "ELT_PROBE_DB"


# ------------------------------------------------------------------ Postgres
@pytest.fixture
def pg_reader():
    cfg = {
        "host": "localhost",
        "port": 5433,
        "database": "address",
        "user": "postgres",
        "password": "testelt",
    }
    reader = TableReader(PostgresConnector(cfg), "address")
    try:
        reader.verify_schema("address", "public")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Local PostgreSQL is unavailable: {exc}")
    yield reader
    reader.close()


def test_pg_fetch_table(pg_reader):
    df = pg_reader.fetch_table("address", "public", "area_code")
    assert df.shape == (53796, 2)
    assert list(df.columns) == ["zip_code", "area_code"]


def test_pg_list_tables(pg_reader):
    tables = pg_reader.list_tables("address", "public")
    assert "area_code" in tables
    assert tables == sorted(tables) or len(tables) >= 4


def test_pg_verify_schema(pg_reader):
    assert pg_reader.verify_schema("address", "public") is True
    assert pg_reader.verify_schema("address", "no_such_schema") is False


# ------------------------------------------------------------------ Snowflake
@pytest.fixture
def sf_reader():
    if not SF_CRED.is_file():
        pytest.skip("Snowflake credential file is missing")
    cfg = json.loads(SF_CRED.read_text())
    if not cfg.get("account") or not cfg.get("password"):
        pytest.skip("Snowflake credentials are empty")
    try:
        from db_connectors import SnowflakeConnector  # type: ignore

        conn = SnowflakeConnector({**cfg, "role": cfg.get("role", "AIRBYTE_ROLE")})
        conn.conn()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Snowflake is unavailable: {type(exc).__name__}: {str(exc)[:80]}")

    cur = conn.conn().cursor()
    try:
        cur.execute(f"CREATE DATABASE IF NOT EXISTS {SF_PROBE_DB}")
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {SF_PROBE_DB}.AIRBYTE_SCHEMA")
        cur.execute(
            f"CREATE OR REPLACE TABLE {SF_PROBE_DB}.AIRBYTE_SCHEMA.states "
            "(abbr string, n int)"
        )
        cur.execute(
            f"INSERT INTO {SF_PROBE_DB}.AIRBYTE_SCHEMA.states VALUES ('AA',1),('CA',2)"
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(
            f"Snowflake writes are unavailable (CREATE DATABASE may be required): "
            f"{str(exc)[:100]}"
        )
    yield TableReader(conn, SF_PROBE_DB)
    try:
        cur.execute(f"DROP DATABASE IF EXISTS {SF_PROBE_DB}")
    except Exception:  # noqa: BLE001
        pass


def test_sf_fetch_table(sf_reader):
    records = sf_reader.fetch_table(SF_PROBE_DB, "AIRBYTE_SCHEMA", "states").to_dict(
        "records"
    )
    assert len(records) == 2
    assert {str(r.get("ABBR") or r.get("abbr")) for r in records} == {"AA", "CA"}


def test_sf_list_tables(sf_reader):
    assert "STATES" in sf_reader.list_tables(SF_PROBE_DB, "AIRBYTE_SCHEMA")


def test_sf_verify_schema(sf_reader):
    assert sf_reader.verify_schema(SF_PROBE_DB, "AIRBYTE_SCHEMA") is True
    assert sf_reader.verify_schema(SF_PROBE_DB, "no_such_schema") is False
