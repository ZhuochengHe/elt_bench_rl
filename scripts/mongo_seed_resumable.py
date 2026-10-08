#!/usr/bin/env python3
"""Resume local MongoDB seeding for ELT-Bench tasks."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import pandas as pd
from pymongo import MongoClient

MONGO_URI = (
    "mongodb://localhost:27017/?directConnection=true&serverSelectionTimeoutMS=3000"
)


def already_done(client: MongoClient, db: str, tables: list[str]) -> bool:
    if db not in client.list_database_names():
        return False
    handle = client[db]
    for table in tables:
        try:
            if handle[table].estimated_document_count() <= 0:
                return False
        except Exception:
            return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--path",
        default=".",
        help="ELT-Bench setup directory containing mongo.jsonl and data/",
    )
    ap.add_argument(
        "--only", nargs="*", default=None, help="Seed only these task databases"
    )
    ap.add_argument(
        "--force", action="store_true", default=os.environ.get("FORCE") == "1"
    )
    args = ap.parse_args()

    setup_dir = Path(args.path).expanduser().resolve()
    manifest = json.loads((setup_dir / "mongo.jsonl").read_text())
    client = MongoClient(MONGO_URI)

    targets = list(manifest.items())
    if args.only:
        wanted = set(args.only)
        targets = [(d, t) for d, t in targets if d in wanted]

    total = len(targets)
    done = skipped = failed = 0
    print(f"==== Seeding {total} databases; force={args.force} ====", flush=True)

    for index, (db, tables) in enumerate(targets, start=1):
        if not args.force and already_done(client, db, tables):
            skipped += 1
            print(f"[SKIP] {db} (already exists and contains documents)", flush=True)
            continue

        print(f"########## [{index}/{total}] {db} ##########", flush=True)
        try:
            client.drop_database(db)
            handle = client[db]
            for table in tables:
                csv_path = setup_dir / "data" / db / f"{table}.csv"
                if not csv_path.is_file():
                    print(
                        f"  [WARN] Missing {csv_path}; skipping collection", flush=True
                    )
                    continue
                df = pd.read_csv(csv_path)
                df = df.where(pd.notnull(df), None)
                rows = [
                    {
                        k: v
                        for k, v in rec.items()
                        if v is not None
                        and not (isinstance(v, float) and math.isnan(v))
                    }
                    for rec in df.to_dict(orient="records")
                ]
                if rows:
                    handle[table].insert_many(rows)

                if db == "card_games" and table == "cards":
                    handle[table].update_many(
                        {"loyalty": {"$type": "number"}},
                        [{"$set": {"loyalty": {"$toString": "$loyalty"}}}],
                    )
            done += 1
            print(f"[OK]   {db}", flush=True)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[FAIL] {db}: {type(exc).__name__}: {exc}", flush=True)

    print(
        f"==== Finished: inserted {done} | skipped {skipped} | failed {failed} ====",
        flush=True,
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
