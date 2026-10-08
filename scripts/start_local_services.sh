#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_COMPOSE="$ROOT_DIR/repo/elt-docker/docker-compose.yml"
LOCAL_COMPOSE="$ROOT_DIR/docker/elt-bench.compose.yml"

if [[ ! -f "$BASE_COMPOSE" ]]; then
  echo "ELT-Bench submodule is missing. Clone with --recurse-submodules or run git submodule update --init --recursive." >&2
  exit 1
fi

docker compose \
  --project-name elt-docker \
  --file "$BASE_COMPOSE" \
  --file "$LOCAL_COMPOSE" \
  up -d "$@"
