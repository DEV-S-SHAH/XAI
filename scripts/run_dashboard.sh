#!/usr/bin/env bash
# ==============================================================================
# Script: run_dashboard.sh
# Launches the interactive Streamlit demonstration dashboard
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

if [ -d "cedge_xai_env" ]; then
    STREAMLIT_EXEC="./cedge_xai_env/bin/streamlit"
else
    STREAMLIT_EXEC="streamlit"
fi

PORT="${PORT:-8501}"

echo "=========================================================="
echo "Launching CEdge-XAI Streamlit Dashboard on port ${PORT}..."
echo "Open browser: http://localhost:${PORT}"
echo "=========================================================="

${STREAMLIT_EXEC} run dashboard/app.py --server.port "${PORT}" --server.headless false
