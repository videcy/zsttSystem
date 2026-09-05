#!/usr/bin/env bash
# POSIX counterpart of start_all.bat: bring up the whole stack on Linux/macOS.
#
#   ./run_all.sh              # containers for Chroma + Neo4j, API from .venv
#   ./run_all.sh --docker     # everything in containers, including the API
#   ./run_all.sh --pipeline   # run the offline pipeline before starting
set -euo pipefail

cd "$(dirname "$0")"

MODE="local"
RUN_PIPELINE=0
for argument in "$@"; do
  case "$argument" in
    --docker) MODE="docker" ;;
    --pipeline) RUN_PIPELINE=1 ;;
    -h|--help)
      sed -n '2,7p' "$0"
      exit 0
      ;;
    *)
      echo "unknown option: $argument" >&2
      exit 2
      ;;
  esac
done

if [[ ! -f .env ]]; then
  echo "missing .env -- copy .env.example to .env and fill in the secrets" >&2
  exit 1
fi

if [[ "$MODE" == "docker" ]]; then
  if [[ "$RUN_PIPELINE" == "1" ]]; then
    docker compose --profile pipeline run --rm pipeline
  fi
  docker compose up -d --wait chromadb neo4j api
  echo "zsttSystem:   http://127.0.0.1:8000"
  echo "ChromaDB:     http://127.0.0.1:8001"
  echo "Neo4j 浏览器: http://127.0.0.1:7474"
  exit 0
fi

if [[ ! -x .venv/bin/python ]]; then
  echo "missing .venv -- run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi
PYTHON=.venv/bin/python

docker compose up -d --wait chromadb neo4j

if [[ "$RUN_PIPELINE" == "1" ]]; then
  "$PYTHON" run_pipeline.py all
fi

if curl -sf -m 3 http://127.0.0.1:8000/health >/dev/null 2>&1; then
  echo "zsttSystem API is already running at http://127.0.0.1:8000"
  exit 0
fi

"$PYTHON" -m uvicorn src.main:app --host 127.0.0.1 --port 8000 &
API_PID=$!
trap 'kill "$API_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 45); do
  if curl -sf -m 2 http://127.0.0.1:8000/health >/dev/null 2>&1; then
    trap - EXIT
    echo "zsttSystem:   http://127.0.0.1:8000 (pid $API_PID)"
    echo "ChromaDB:     http://127.0.0.1:8001"
    echo "Neo4j 浏览器: http://127.0.0.1:7474"
    exit 0
  fi
  sleep 1
done

echo "API did not become healthy within 45s" >&2
exit 1
