"""Exercise reward grading against a local PostgreSQL instance."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "repo" / "evaluation"))

import pandas as pd  # noqa: E402
import psycopg2  # noqa: E402

from eltbench.destinations import PostgresConnector  # noqa: E402
from eltbench.reward import compute_reward  # noqa: E402

TASK = "address"
GT = ROOT / "repo" / "evaluation" / "gt" / TASK
PG = {"host": "localhost", "port": 5433, "user": "postgres", "password": "testelt"}
PROBE_DB = "elt_reward_probe"


def qc(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def materialize(conn, csv_path: Path) -> None:
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    table = csv_path.stem
    collist = ", ".join(qc(c) for c in df.columns)
    conn.execute(f"DROP TABLE IF EXISTS public.{qc(table)}")
    conn.execute(
        f"CREATE TABLE public.{qc(table)} "
        f"({', '.join(qc(c) + ' text' for c in df.columns)})"
    )
    ph = ", ".join(["%s"] * len(df.columns))
    cur = conn.conn().cursor()
    cur.executemany(
        f"INSERT INTO public.{qc(table)} ({collist}) VALUES ({ph})",
        [tuple(r) for r in df.itertuples(index=False, name=None)],
    )
    cur.close()


def score(conn) -> str:
    r = compute_reward(conn, TASK, GT, PROBE_DB, "public", load_tables=[])
    return f"model={r.model_score:.3f} total={r.total:.4f}"


def main() -> int:
    admin = psycopg2.connect(**PG, dbname="postgres")
    admin.autocommit = True
    cur = admin.cursor()
    cur.execute(
        "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
        f"WHERE datname = '{PROBE_DB}'"
    )
    cur.execute(f'DROP DATABASE IF EXISTS "{PROBE_DB}"')
    cur.execute(f'CREATE DATABASE "{PROBE_DB}"')
    cur.close()

    conn = PostgresConnector({**PG, "database": PROBE_DB})
    conn.reset_namespace(PROBE_DB, "public")
    tables = sorted(GT.glob("*.csv"))
    print(f"Task {TASK}: {len(tables)} data models -> {[t.stem for t in tables]}")

    failures: list[str] = []

    def check(title: str, expect_model: float) -> None:
        got = score(conn)
        ok = f"model={expect_model:.3f}" in got
        if not ok:
            failures.append(title)
        print(f"  {'✅' if ok else '❌'} {title:24} {got}")

    for t in tables:
        materialize(conn, t)
    check("Ground truth", 1.0)

    conn.execute(
        f"DELETE FROM public.{qc(tables[0].stem)} "
        f"WHERE ctid IN (SELECT ctid FROM public.{qc(tables[0].stem)} LIMIT 1)"
    )
    check("One row missing", 0.0)

    materialize(conn, tables[0])
    conn.execute(
        f"UPDATE public.{qc(tables[0].stem)} "
        f"SET {qc('NUM_COUNTIES')} = "
        f"(COALESCE({qc('NUM_COUNTIES')}, '0')::numeric * 1.10)::text"
    )
    check("One column +10%", 0.0)

    conn.execute(
        f"ALTER TABLE public.{qc(tables[0].stem)} "
        f"RENAME COLUMN {qc('NUM_COUNTIES')} TO {qc('num_counties_renamed')}"
    )
    check("Column renamed", 0.0)

    materialize(conn, tables[0])
    conn.execute(f"UPDATE public.{qc(tables[0].stem)} SET {qc('NUM_COUNTIES')} = NULL")
    check("Column set to NULL", 0.0)

    materialize(conn, tables[0])
    conn.execute(
        f"CREATE TABLE public.tmp_rev AS SELECT * FROM public.{qc(tables[0].stem)}"
    )
    conn.execute(f"DROP TABLE public.{qc(tables[0].stem)}")
    conn.execute(f"ALTER TABLE public.tmp_rev RENAME TO {qc(tables[0].stem)}")
    check("Rows reversed (same values)", 1.0)

    cur = admin.cursor()
    cur.execute(
        "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
        f"WHERE datname = '{PROBE_DB}'"
    )
    cur.execute(f'DROP DATABASE IF EXISTS "{PROBE_DB}"')
    cur.close()
    admin.close()
    conn.close()

    print()
    if failures:
        print(f"{len(failures)} cases did not match expectations: {failures}")
        return 1
    print("All cases match expectations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
