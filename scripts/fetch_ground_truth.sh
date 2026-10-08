#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSET_DIR="$ROOT_DIR/runs/benchmark-assets"
EVALUATION_DIR="$ASSET_DIR/evaluation"
HF_DIR="$EVALUATION_DIR/huggingface"
GT_DIR="$EVALUATION_DIR/gt"
DESTINATION="${1:-snowflake}"

case "$DESTINATION" in
  local_postgres|snowflake) HF_VARIANT="snowflake" ;;
  databricks|redshift) HF_VARIANT="$DESTINATION" ;;
  *)
    echo "Usage: $0 [local_postgres|snowflake|databricks|redshift]" >&2
    exit 2
    ;;
esac

HF_GT_DIR="$HF_DIR/gt_$HF_VARIANT"

if ! command -v hf >/dev/null 2>&1; then
  echo "Hugging Face CLI is missing. Activate the project environment and install dependencies with: uv pip install -r requirements.txt" >&2
  exit 1
fi

mkdir -p "$HF_DIR" "$GT_DIR"
hf download tttjjj/elt_bench --repo-type dataset \
  --include "gt_$HF_VARIANT/**" --local-dir "$HF_DIR"

if [[ ! -d "$HF_GT_DIR" ]] || ! find "$HF_GT_DIR" -mindepth 2 -maxdepth 2 -type f -name '*.csv' -print -quit | grep -q .; then
  echo "Hugging Face download did not produce task CSV files under $HF_GT_DIR" >&2
  exit 1
fi

cp -a "$HF_GT_DIR/." "$GT_DIR/"
printf 'Ground truth downloaded to: %s\n' "$GT_DIR"
