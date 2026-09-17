#!/usr/bin/env bash
# ==============================================================================
# Script: run_api.sh
# Starts the CEdge-XAI FastAPI Microservice Daemon
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

if [ -d "cedge_xai_env" ]; then
    PYTHON_EXEC="./cedge_xai_env/bin/python"
    UVICORN_EXEC="./cedge_xai_env/bin/uvicorn"
else
    PYTHON_EXEC="python3"
    UVICORN_EXEC="uvicorn"
fi

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

echo "=========================================================="
echo "Starting CEdge-XAI FastAPI Service on http://${HOST}:${PORT}"
echo "API Docs: http://${HOST}:${PORT}/docs"
echo "=========================================================="

${UVICORN_EXEC} api.main:app --host "${HOST}" --port "${PORT}" --workers 1
