#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSET_DIR="$ROOT_DIR/runs/benchmark-assets/upstream"

if ! command -v gdown >/dev/null 2>&1; then
  echo "gdown is missing. Install project dependencies with: uv pip install -r requirements.txt" >&2
  exit 1
fi

fetch_and_extract() {
  local name="$1"
  local file_id="$2"
  local destination="$3"
  local archive="$ASSET_DIR/$name.zip"
  local marker="$destination/data/.eltbench-extracted"

  if [[ -f "$marker" ]]; then
    printf 'Already extracted: %s\n' "$destination/data"
    return
  fi

  mkdir -p "$ASSET_DIR" "$destination"
  gdown --continue "https://drive.google.com/uc?id=$file_id" -O "$archive"
  unzip -tq "$archive"
  unzip -oq "$archive" -d "$destination"
  if [[ ! -d "$destination/data" ]] || ! find "$destination/data" -mindepth 2 -type f -print -quit | grep -q .; then
    echo "Archive $archive did not produce data files under $destination/data" >&2
    exit 1
  fi
  touch "$marker"
  printf 'Extracted %s to %s/data\n' "$name" "$destination"
}

fetch_and_extract data_api 1qVAzU3kgn_G72QQ4b5zt3e1hwkQcSgDq \
  "$ROOT_DIR/repo/elt-docker/rest_api"
fetch_and_extract data_db 1-Gv5g_Yg_YrR-NxH4s2tSEK3VhJQc2Q0 \
  "$ROOT_DIR/repo/setup"
