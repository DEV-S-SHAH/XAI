#!/usr/bin/env bash
# ==============================================================================
# Script: run_federated.sh
# Executes the Flower FedAvg multi-factory edge simulation
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

if [ -d "xai_env" ]; then
    PYTHON_EXEC="./xai_env/bin/python"
elif [ -d "cedge_xai_env" ]; then
    PYTHON_EXEC="./cedge_xai_env/bin/python"
elif [ -d "venv" ]; then
    PYTHON_EXEC="./venv/bin/python"
else
    PYTHON_EXEC="python3"
fi

ROUNDS="${1:-5}"
LOCAL_EPOCHS="${2:-2}"

echo "=========================================================="
echo "Running Decentralized Federated Learning Simulation"
echo "Nodes: 3 Edge Factories (A, B, C) | Strategy: FedAvg"
echo "Communication Rounds: ${ROUNDS} | Local Epochs: ${LOCAL_EPOCHS}"
echo "=========================================================="

${PYTHON_EXEC} models/federated/fl_server.py --rounds "${ROUNDS}"

echo "Federated convergence saved to paper_assets/tables/fl_convergence.csv"
