#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSET_DIR="$ROOT_DIR/runs/benchmark-assets"
ARCHIVE="$ASSET_DIR/gt.zip"
EVALUATION_DIR="$ASSET_DIR/evaluation"

if ! command -v gdown >/dev/null 2>&1; then
  echo "gdown is missing. Install project dependencies with: uv pip install -r requirements.txt" >&2
  exit 1
fi

mkdir -p "$ASSET_DIR" "$EVALUATION_DIR"
gdown 'https://drive.google.com/uc?id=11vQqNEWXoPG6sjKytAn7TtFLDMiQa17I' -O "$ARCHIVE"
unzip -q -o "$ARCHIVE" -d "$EVALUATION_DIR"

GT_DIR="$EVALUATION_DIR/gt"
if [[ ! -d "$GT_DIR" ]] || ! find "$GT_DIR" -mindepth 2 -maxdepth 2 -type f -name '*.csv' -print -quit | grep -q .; then
  echo "Ground-truth archive did not produce task CSV files in $GT_DIR" >&2
  exit 1
fi

printf 'Ground truth downloaded to: %s\n' "$GT_DIR"
