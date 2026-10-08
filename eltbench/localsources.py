"Materialize non-PostgreSQL sources for local rollouts."

from __future__ import annotations

import io
import json
import subprocess
import urllib.request
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "runs" / "_src_cache"
RAW_SEED_DIR = PROJECT_ROOT / "repo" / "setup" / "data"


LOCALSTACK_URL = "http://localhost:4566"
ELT_API_URL = "http://localhost:5005"
MONGO_CONTAINER = "elt-mongodb"


def _infer_type(values: Iterable[Any]) -> str:
    seen = {type(v) for v in values if v is not None}
    if not seen or seen <= {bool}:
        return "boolean" if seen else "text"
    if seen <= {int}:
        return "bigint"
    if seen <= {int, float}:
        return "double precision"
    return "text"


def _write_table(conn, schema: str, table: str, rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    types = {c: _infer_type([r.get(c) for r in rows]) for c in cols}
    with conn.cursor() as cur:
        cur.execute(f'DROP TABLE IF EXISTS "{schema}"."{table}" CASCADE')
        ddl = ", ".join(f'"{c}" {types[c]}' for c in cols)
        cur.execute(f'CREATE TABLE "{schema}"."{table}" ({ddl})')
        buf = io.StringIO()
        import csv as _csv

        writer = _csv.writer(buf)
        for r in rows:
            writer.writerow(
                [
                    json.dumps(r.get(c))
                    if isinstance(r.get(c), (dict, list))
                    else r.get(c)
                    for c in cols
                ]
            )
        buf.seek(0)
        cur.copy_expert(f'COPY "{schema}"."{table}" FROM STDIN WITH CSV NULL \'\'', buf)
    return len(rows)


def _rows_from_raw_seed(task: str, table: str) -> list[dict[str, Any]] | None:
    import pandas as pd

    for ext in (".jsonl", ".csv", ".json"):
        path = RAW_SEED_DIR / task / f"{table}{ext}"
        if not path.is_file():
            continue
        if ext == ".jsonl":
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            frame = pd.read_csv(path, dtype=object, keep_default_na=False)
            rows = frame.to_dict(orient="records")
        return rows if isinstance(rows, list) else None
    return None


def _rows_from_custom_api(task: str, table: str) -> list[dict[str, Any]]:
    url = f"{ELT_API_URL}/{task}/{table}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = json.loads(resp.read())
    return data if isinstance(data, list) else data.get("data", [])


def _rows_from_s3(path: str) -> list[dict[str, Any]]:
    import boto3

    assert path.startswith("s3://"), path
    bucket, _, key = path[5:].partition("/")
    client = boto3.client(
        "s3",
        endpoint_url=LOCALSTACK_URL,
        aws_access_key_id="test",
        aws_secret_access_key="test",
        region_name="us-west-2",
    )
    raw = (
        client.get_object(Bucket=bucket, Key=key)["Body"]
        .read()
        .decode("utf-8", "replace")
    )
    rows = []
    for line in raw.splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _rows_from_mongo(
    connection_string: str, database: str, table: str
) -> list[dict[str, Any]]:
    code = (
        f'JSON.stringify(db.getSiblingDB("{database}").getCollection("{table}")'
        f".find().toArray())"
    )
    proc = subprocess.run(
        ["docker", "exec", MONGO_CONTAINER, "mongosh", "--quiet", "--eval", code],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip()[:200])
    rows = json.loads(proc.stdout.strip() or "[]")
    for r in rows:
        r.pop("_id", None)
    return rows


def _rows_from_flat_file(task: str, url: str, table: str) -> list[dict[str, Any]]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = CACHE_DIR / f"{task}__{table}.csv"
    if not cached.is_file():
        try:
            with urllib.request.urlopen(url, timeout=600) as resp:
                cached.write_bytes(resp.read())
        except Exception:
            import shutil as _shutil
            import sys as _sys

            gdown = _shutil.which("gdown") or str(
                Path(_sys.executable).parent / "gdown"
            )
            subprocess.run(
                [gdown, url, "-O", str(cached)],
                check=True,
                capture_output=True,
                text=True,
                timeout=600,
            )
    import pandas as pd

    frame = pd.read_csv(cached, dtype=object, keep_default_na=False)
    return frame.to_dict(orient="records")


def materialize_non_postgres_sources(
    config: dict,
    *,
    target_db: str,
    target_schema: str,
    task: str,
    harness: dict[str, Any],
) -> dict[str, int]:
    import psycopg2

    plans: list[tuple[str, callable]] = []
    for spec in (config.get("aws_s3", {}) or {}).get("data", []) or []:
        plans.append((spec["table"], lambda s=spec: _rows_from_s3(s["path"])))
    api = (config.get("custom_api", {}) or {}).get("config", {}) or {}
    for t in api.get("tables", []) or []:
        plans.append((t, lambda t=t: _rows_from_custom_api(task, t)))
    mongo = (config.get("mongodb", {}) or {}).get("config", {}) or {}
    for t in mongo.get("tables", []) or []:
        plans.append(
            (
                t,
                lambda t=t, m=mongo: _rows_from_mongo(
                    m.get("connection_string", ""), m.get("database", task), t
                ),
            )
        )
    for spec in config.get("flat_files", []) or []:
        plans.append(
            (
                spec["table"],
                lambda s=spec: _rows_from_flat_file(task, s["path"], s["table"]),
            )
        )

    conn = psycopg2.connect(
        host=harness["host"],
        port=int(harness["port"]),
        user=harness["user"],
        password=harness["password"],
        dbname=target_db,
    )
    conn.autocommit = True
    written: dict[str, int] = {}
    failures: list[str] = []
    try:
        for table, fetch in plans:
            try:
                rows = _rows_from_raw_seed(task, table)
                if rows is None:
                    rows = fetch()
                written[table] = _write_table(conn, target_schema, table, rows)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{table}: {type(exc).__name__}: {str(exc)[:160]}")
    finally:
        conn.close()
    if failures:
        raise RuntimeError(
            f"Failed to materialize non-PostgreSQL sources for task {task}: "
            + "; ".join(failures)
        )
    return written
