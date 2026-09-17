#!/usr/bin/env bash
# ==============================================================================
# Script: run_training.sh
# End-to-end training and evaluation of all models and XAI discovery
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

if [ -d "cedge_xai_env" ]; then
    PYTHON_EXEC="./cedge_xai_env/bin/python"
else
    PYTHON_EXEC="python3"
fi

echo "=========================================================="
echo "Starting CEdge-XAI Full Training & Discovery Pipeline"
echo "=========================================================="

echo "[1/7] Generating Synthetic IoT Telemetry Dataset (4320 rows, 10 features)..."
${PYTHON_EXEC} data/synthetic/generate_dataset.py

echo "[2/7] Training All 4 Detection Architectures..."
${PYTHON_EXEC} models/detection/train.py

echo "[3/7] Running Benchmark Evaluation & Comparison (Tables & Figures)..."
${PYTHON_EXEC} models/detection/compare_models.py

echo "[4/7] Discovering Causal Directed Graph (Tigramite PCMCI)..."
${PYTHON_EXEC} models/xai/causal_graph.py

echo "[5/7] Verifying Counterfactual Explanations..."
${PYTHON_EXEC} models/xai/counterfactual_engine.py

echo "[6/7] Exporting ONNX & Quantizing INT8..."
${PYTHON_EXEC} models/detection/export_onnx.py
${PYTHON_EXEC} models/detection/quantize.py

echo "[7/7] Benchmarking Edge Runtimes (1000 iterations)..."
${PYTHON_EXEC} experiments/edge_benchmark.py

echo "=========================================================="
echo "CEdge-XAI Training Pipeline Complete!"
echo "Checkpoints saved in models_saved/checkpoints/"
echo "Assets saved in paper_assets/tables/ and paper_assets/figures/"
echo "=========================================================="
