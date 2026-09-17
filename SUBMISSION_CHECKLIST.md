# XAI for IOT Anomaly Detection Submission Checklist

This document verifies the completion and validation status of all required deliverables for the **XAI for IOT Anomaly Detection** project.

---

## Deliverables Status

- [x] **Deliverable 1: Complete Source Code**
  - Canonical project structure rooted in `models/`, `api/`, `dashboard/`, `experiments/`, `config/`, and `data/`.
  - Zero redundant legacy folders (`src/`, `configs/`, `notebooks/` removed).
  - All code fully compliant with PEP 8, formatted with Black and Flake8.

- [x] **Deliverable 2: Synthetic Dataset**
  - Location: `data/synthetic/factory_iot_data.csv`
  - Dimensions: Exactly 4,320 rows (3 days at 1-min sampling rate) and 10 continuous telemetry features.
  - Failure Scenarios: A1 (Fan Failure), A2 (Lubrication Leak), A3 (Heatwave), A4 (Sensor Glitch), A5 (Spindle Overload).
  - Anomaly Ratio: ~4.0% (173 anomaly minutes).

- [x] **Deliverable 3: Trained Model Checkpoints**
  - Location: `models_saved/checkpoints/`
  - LSTM-Autoencoder (`lstm_ae_best.pt`)
  - Isolation Forest (`isolation_forest.joblib`)
  - One-Class SVM (`one_class_svm.joblib`)
  - Anomaly Transformer (`anomaly_transformer_best.pt`)
  - Normalization Scaler (`scaler.joblib`)

- [x] **Deliverable 4: ONNX and INT8 Quantized Models**
  - Location: `models_saved/onnx/` and `models_saved/quantized/`
  - ONNX Model: `models_saved/onnx/lstm_autoencoder.onnx` (FP32 graph)
  - INT8 Model: `models_saved/quantized/lstm_autoencoder_int8.onnx` (Quantized edge deployment)
  - Verified edge acceleration with sub-5ms inference latency.

- [x] **Deliverable 5: All 9 Result Tables as CSV**
  - Location: `paper_assets/tables/`
  1. `dataset_summary.csv` — Feature distributions, units, physical baselines
  2. `model_comparison.csv` — Comprehensive benchmark across 4 detectors (F1, AUC, Latency, Size)
  3. `xai_comparison.csv` — Evaluation across Counterfactuals, Causal Graph, and SHAP
  4. `counterfactual_accuracy.csv` — Root-cause attribution accuracy across failure scenarios
  5. `causal_accuracy.csv` — Structural Hamming Distance, precision, and recall on causal discovery
  6. `fl_convergence.csv` — 5-round FedAvg convergence trajectory across 3 edge factories
  7. `edge_benchmark.csv` — Hardware profiling across PyTorch CPU, ONNX, and INT8 Quantized
  8. `ablation_study.csv` — Impact of feature count, window length, and loss formulation
  9. `real_validation.csv` — Empirical validation on the AI4I 2020 Predictive Maintenance dataset

- [x] **Deliverable 6: All 15 Figures as PNG**
  - Location: `paper_assets/figures/` (Publication-grade 300 DPI, 50KB–2MB each)
  1. `architecture.png` — XAI for IOT Anomaly Detection Split-Brain dual-engine architecture
  2. `dataset_overview.png` — Multivariate telemetry across normal and failure windows
  3. `roc_curves.png` — Receiver Operating Characteristic curves across all 4 models
  4. `pr_curves.png` — Precision-Recall curves across all 4 models
  5. `f1_comparison.png` — F1 score comparison bar chart
  6. `confusion_matrix.png` — Anomaly detection confusion matrix
  7. `latency_vs_size.png` — Edge deployment trade-off scatter plot
  8. `causal_graph.png` — Learned directed temporal causal sensor topology (PCMCI)
  9. `shap_summary.png` — Feature importance attribution summary
  10. `xai_comparison.png` — Runtime latency comparison across XAI methods
  11. `fl_convergence.png` — Federated validation loss trajectory over rounds
  12. `fl_vs_centralized.png` — Federated vs. centralized performance parity
  13. `edge_latency.png` — Latency profiling across PyTorch, ONNX, and INT8
  14. `edge_size.png` — Memory footprint compression across execution runtimes
  15. `ablation_study.png` — Ablation study sensitivity analysis curves

- [x] **Deliverable 7: Research Paper in LaTeX**
  - Location: `paper/main.tex`
  - Formatted for IEEE Transactions on Industrial Informatics (double column).
  - Complete with abstract, introduction, methodology, experiments, results, and references.

- [x] **Deliverable 8: Docker Configuration for Reproducibility**
  - `Dockerfile` — Multi-stage slim container supporting FastAPI, Streamlit, and Flower FL.
  - `docker-compose.yml` — Multi-service orchestration (`edge-server`, `dashboard`, `federated-simulation`).
  - `.dockerignore` — Excludes local environments, caches, and raw artifacts.

- [x] **Deliverable 9: Production README**
  - Location: `README.md`
  - Formatted strictly according to Phase 3J with 9 clear sections: Title, Architecture, Tech Stack, Installation, Project Structure, Execution Guide, Results Table, BibTeX Citation, and License.

- [x] **Deliverable 10: Test Suite with 100% Pass Rate**
  - Location: `tests/`
  - `test_model.py` — Unit tests for all 4 detection models and ONNX runtime.
  - `test_xai.py` — Unit tests for Counterfactual Engine and Causal Graph discovery.
  - `test_data.py` — Unit tests for data loading, sliding windows, and real dataset preprocessing.
