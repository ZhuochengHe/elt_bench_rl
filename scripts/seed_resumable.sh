#!/usr/bin/env bash
# Resume ELT-Bench source seeding by skipping tasks that are already populated.
# Usage:
#   bash scripts/seed_resumable.sh
#   FORCE=1 bash scripts/seed_resumable.sh
#   bash scripts/seed_resumable.sh address airline

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SETUP_DIR="${SETUP_DIR:-$ROOT_DIR/repo/setup}"
LOG="${LOG:-$ROOT_DIR/runs/seed_resumable.log}"
PG_PORT="${PG_PORT:-5433}"
PG_USER="${PG_USER:-postgres}"
PG_PASS="${PG_PASS:-testelt}"
S3_ENDPOINT="${S3_ENDPOINT:-http://localhost:4566}"

export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-west-2}"
export AWS_REQUEST_CHECKSUM_CALCULATION=when_required
export PGPASSWORD="$PG_PASS"

mkdir -p "$(dirname "$LOG")"

psql_q() { psql -h localhost -U "$PG_USER" -p "$PG_PORT" -tAc "$1" 2>/dev/null; }

# Check whether the database already contains public tables.
pg_done_db() {
  local db="$1"
  local n
  n=$(psql -h localhost -U "$PG_USER" -p "$PG_PORT" -d "$db" -tAc \
        "SELECT count(*) FROM information_schema.tables WHERE table_schema='public'" 2>/dev/null)
  [ -n "${n:-}" ] && [ "$n" -gt 0 ] 2>/dev/null
}

s3_done() {
  aws --endpoint-url="$S3_ENDPOINT" s3 ls "s3://$1-bucket" >/dev/null 2>&1
}

cd "$SETUP_DIR"
CURRENT_DIR="$(pwd)"

# Build the list of tasks to process.
if [ "$#" -gt 0 ]; then
  todo=("$@")
else
  todo=()
  for d in "$CURRENT_DIR"/sources/*/; do
    todo+=("$(basename "$d")")
  done
fi

total=${#todo[@]}
done_cnt=0; skip_cnt=0; fail_cnt=0
echo "==================== Seeding started $(date) ====================" | tee -a "$LOG"
echo "Tasks to check: $total; FORCE=${FORCE:-0}" | tee -a "$LOG"

for db in "${todo[@]}"; do
  if [ "${FORCE:-0}" != "1" ]; then
    if pg_done_db "$db" && s3_done "$db"; then
      skip_cnt=$((skip_cnt+1))
      echo "[SKIP] $db (PostgreSQL tables and S3 bucket already exist)" | tee -a "$LOG"
      continue
    fi
  fi

  echo "########## [$((done_cnt+1))/$total] $db ##########" | tee -a "$LOG"
  if ( cd "sources/$db" && bash data.sh "$CURRENT_DIR" ) >>"$LOG" 2>&1; then
    done_cnt=$((done_cnt+1))
    echo "[OK]   $db" | tee -a "$LOG"
  else
    fail_cnt=$((fail_cnt+1))
    echo "[FAIL] $db (see $LOG)" | tee -a "$LOG"
  fi
done

echo "==================== Finished $(date) ====================" | tee -a "$LOG"
echo "Processed: $done_cnt | Skipped: $skip_cnt | Failed: $fail_cnt" | tee -a "$LOG"
