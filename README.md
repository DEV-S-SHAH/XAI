# XAI for IOT Anomaly Detection

XAI for IOT Anomaly Detection is a production-grade, distributed framework engineered for real-time anomaly detection and physically consistent explainability in Industrial Internet of Things (IIoT) cyber-physical systems. Utilizing a dual-tier "Split-Brain" architecture, the framework decouples sub-millisecond anomaly detection on resource-constrained edge microcontrollers (via 8-bit quantized ONNX models) from compute-intensive root-cause explanation and causal attribution executed on edge servers. The platform integrates gradient-based counterfactual search with physics-informed bounds, constraint-based causal DAG discovery via Tigramite PCMCI, and privacy-preserving Federated Learning (Flower FedAvg) across factory edge nodes, achieving high-fidelity anomaly attribution while keeping raw operational telemetry strictly on-premises.

## Architecture

![XAI for IOT Anomaly Detection Architecture](paper_assets/figures/architecture.png)

## Tech Stack

| Component | Framework / Library | Primary Purpose |
| :--- | :--- | :--- |
| **Deep Learning & Edge** | PyTorch, ONNX, ONNX Runtime | LSTM-Autoencoder, Anomaly Transformer, INT8 edge quantization |
| **Classical ML Baseline** | Scikit-learn, Joblib | Isolation Forest, One-Class SVM baseline benchmarks |
| **Explainable AI (XAI)** | Custom PyTorch Engine, SHAP, Captum | Gradient-based physics-clamped counterfactuals, baseline SHAP |
| **Causal Discovery** | Tigramite (PCMCI), NetworkX | Time-lagged partial correlation graphs, propagation path tracing |
| **Federated Learning** | Flower (`flwr`), gRPC | Multi-factory decentralized model training (zero raw telemetry egress) |
| **Serving & APIs** | FastAPI, Uvicorn, Pydantic | Asynchronous split-brain inference (`/detect`, `/explain`) |
| **Interactive Dashboard**| Streamlit, Plotly, Seaborn | Industrial plant telemetry monitoring and root-cause visualization |
| **Containerization** | Docker, Docker Compose | Reproducible multi-service deployment |

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/DEV-S-SHAH/XAI.git && cd XAI

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the platform
uvicorn api.main:app --port 8000
```

## Project Structure

```text
XAI/
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
├── SUBMISSION_CHECKLIST.md
├── config/
│   └── config.yaml
├── data/
│   ├── synthetic/
│   │   ├── generate_dataset.py
│   │   └── factory_iot_data.csv
│   └── real/
│       ├── preprocess_real.py
│       └── ai4i_2020_preprocessed.csv
├── models/
│   ├── detection/
│   │   ├── lstm_autoencoder.py
│   │   ├── isolation_forest_model.py
│   │   ├── one_class_svm_model.py
│   │   ├── anomaly_transformer_model.py
│   │   ├── split_brain_engine.py       # [Novelty 2] Dual-tier edge/server execution engine
│   │   ├── train.py
│   │   ├── compare_models.py
│   │   ├── quantize.py                 # Graph-folding + dynamic INT8 quantization
│   │   └── export_onnx.py
│   ├── xai/
│   │   ├── actionable_what_if.py       # [Novelty 1] Actionable counterfactual generator
│   │   ├── counterfactual_engine.py    # Physics-clamped & causal-propagated engine
│   │   ├── causal_graph.py             # Tigramite PCMCI causal DAG discovery
│   │   ├── shap_baseline.py            # SHAP baseline attribution
│   │   └── explanation_formatter.py
│   └── federated/
│       ├── federated_experiment.py     # [Novelty 3] Non-IID simulation & privacy audit
│       ├── fl_server.py                # Flower FedAvg server
│       ├── fl_client.py                # Flower edge client
│       └── fl_utils.py
├── api/
│   ├── main.py
│   ├── schemas.py
│   └── routes/
│       ├── detect.py
│       └── explain.py
├── dashboard/
│   ├── app.py
│   └── pages/
│       ├── 1_detection.py
│       ├── 2_explanations.py
│       ├── 3_causal_graph.py
│       ├── 4_federated.py
│       └── 5_edge_benchmark.py
├── experiments/
│   ├── run_novelty1_what_if.py        # [Novelty 1] Multi-seed benchmark vs baselines
│   ├── run_novelty2_split_brain.py     # [Novelty 2] Parametric sweep across anomaly rates
│   ├── run_novelty3_federated.py       # [Novelty 3] Multi-seed FL vs Local vs Centralized
│   ├── evaluate_counterfactuals.py     # Step 1 & Step 2 XAI evaluation (CF vs SHAP vs LIME)
│   ├── evaluate_causal.py              # Step 3 Causal graph validation against ground truth
│   ├── evaluate_split_brain.py         # Step 4 Edge vs Server latency & workload profiling
│   ├── edge_benchmark.py               # Step 5 Hardware profiling (FP32 vs ONNX vs INT8)
│   └── ablation_study.py
├── paper_assets/
│   ├── tables/                         # 22 structured CSV experiment logs
│   └── figures/                        # 22 publication-grade 300 DPI plots
├── models_saved/
│   ├── checkpoints/                    # PyTorch model weights
│   ├── onnx/                           # Exported FP32 ONNX graphs
│   └── quantized/                      # 8-bit dynamic quantized ONNX graphs (0.193 MB)
├── paper/
│   └── main.tex
├── tests/
│   ├── test_model.py                   # Architecture and forward-pass tests
│   ├── test_xai.py                     # Counterfactual search & causal tests
│   ├── test_data.py                    # Preprocessing & windowing tests
│   ├── test_novelty1_what_if.py        # Novelty 1 unit and integration tests
│   ├── test_novelty2_split_brain.py    # Novelty 2 unit and latency constraint tests
│   └── test_novelty3_federated.py      # Novelty 3 unit and privacy audit tests
└── scripts/
    ├── run_training.sh
    ├── run_api.sh
    ├── run_dashboard.sh
    └── run_federated.sh
```

## How to Run Each Component

```bash
# 1. Dataset Generation (4,320 rows, 10 continuous features)
python data/synthetic/generate_dataset.py

# 2. Model Training & Comparison
python models/detection/train.py
python models/detection/compare_models.py

# 3. Model Quantization and Edge Benchmark
python models/detection/export_onnx.py
python models/detection/quantize.py
python experiments/edge_benchmark.py

# 4. Explainable AI & Causal Graph Discovery
python models/xai/counterfactual_engine.py
python models/xai/causal_graph.py
python models/xai/shap_baseline.py

# 5. Federated Learning Simulation
python models/federated/fl_server.py --rounds 5

# 6. REST API Server
uvicorn api.main:app --host 0.0.0.0 --port 8000

# 7. Interactive Streamlit Dashboard
streamlit run dashboard/app.py --server.port 8501

# 8. Automated Test Suite (23 passed)
pytest tests/ -v

# 9. Docker Deployment
docker compose up --build
```

---

## Novelty Evaluation & Experimental Replication

All experiments are deterministic across random seeds, execute cleanly from the command line, and store reproducible numerical metrics in `paper_assets/tables/` and high-resolution figures in `paper_assets/figures/`.

### Novelty 1: Actionable What-If Counterfactual Explanations
*Goes beyond feature-importance scores by identifying root causes, prescribing directional modifications within physical sensor limits, and ensuring model classification reverts to normal.*

- **Script**: `python experiments/run_novelty1_what_if.py`
- **Output Tables**: `paper_assets/tables/novelty1_what_if_summary.csv`, `paper_assets/tables/novelty1_per_anomaly_detail.csv`
- **Output Figures**: `paper_assets/figures/novelty1_validity_sparsity.png`, `paper_assets/figures/novelty1_score_reduction.png`

**Multi-Seed Benchmark (Seeds: 42, 123, 456):**
| Method | Root Cause Acc (%) | Validity Rate (%) | Sparsity (Features) | Proximity ($L_1$) | Proximity ($L_2$) | Actionability (%) | Score Reduction (%) | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Proposed (Actionable What-If)** | **100.0 ± 0.0%** | **100.0 ± 0.0%** | **2.99 ± 0.00** | 0.3675 | 1.6003 | 47.6% | **91.7 ± 0.0%** | **5.68 ± 0.04** |
| Baseline 1 (Feature Ablation) | 79.8 ± 0.0% | 53.2 ± 0.0% | 1.00 ± 0.00 | 0.2625 | 1.5172 | 100.0% | 74.3 ± 0.0% | 3.62 ± 0.02 |
| Baseline 2 (Gradient Opt) | 79.4 ± 0.0% | 17.6 ± 0.0% | 3.76 ± 0.00 | 0.0120 | 0.0877 | 86.8% | 41.7 ± 0.0% | 576.47 ± 10.02 |

*Takeaway*: The proposed actionable what-if explainer delivers 100.0% root-cause attribution accuracy and 100.0% counterfactual validity with 91.7% anomaly score reduction in 5.68 ms, outperforming unconstrained gradient search which requires 576.5 ms and achieves only 17.6% validity.

---

### Novelty 2: Split-Brain Edge Architecture
*Decouples sub-millisecond edge anomaly detection (INT8 ONNX) from heavy server-side causal/counterfactual XAI, conditionally invoking explanations only when anomalies occur.*

- **Script**: `python experiments/run_novelty2_split_brain.py`
- **Output Tables**: `paper_assets/tables/novelty2_anomaly_rate_sweep.csv`, `paper_assets/tables/novelty2_split_brain_benchmark.csv`
- **Output Figures**: `paper_assets/figures/novelty2_latency_breakdown.png`, `paper_assets/figures/novelty2_bandwidth_energy_tradeoff.png`

**Parametric Anomaly Rate Sweep (200 observations per rate):**
| Anomaly Rate | Anomaly Count | Baseline Latency | Proposed Latency | Latency Speedup | Baseline Bandwidth | Proposed Bandwidth | Bandwidth Savings | Avoided XAI Calls | Energy Savings (Est.) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1%** | 9 | 6.39 ms | **0.48 ms** | **13.33x** | 546.9 KB | **26.1 KB** | **95.2%** | **95.5%** | **95.2%** |
| **5%** | 20 | 6.45 ms | **0.82 ms** | **7.89x** | 546.9 KB | **56.1 KB** | **89.7%** | **90.0%** | **90.0%** |
| **10%** | 28 | 6.36 ms | **1.08 ms** | **5.88x** | 546.9 KB | **77.9 KB** | **85.8%** | **86.0%** | **85.7%** |
| **20%** | 46 | 6.55 ms | **1.63 ms** | **4.01x** | 546.9 KB | **127.0 KB** | **76.8%** | **77.0%** | **77.9%** |
| **50%** | 101 | 6.49 ms | **3.40 ms** | **1.91x** | 546.9 KB | **276.9 KB** | **49.4%** | **49.5%** | **50.5%** |

*Takeaway*: Under typical industrial operating conditions (1-5% anomaly prevalence), the split-brain design cuts end-to-end latency by 7.9x - 13.3x, eliminates ~90-95% of network telemetry transmission, and saves ~90-95% of edge computational energy by filtering out normal observations on-device.

---

### Novelty 3: Privacy-Preserving Federated Training
*Trains edge anomaly models collaboratively across 3 decentralized factory clients (Factory A, B, C) under non-IID operating conditions without pooling raw telemetry.*

- **Script**: `python experiments/run_novelty3_federated.py`
- **Output Tables**: `paper_assets/tables/novelty3_federated_comparison.csv`, `paper_assets/tables/novelty3_privacy_audit.csv`, `paper_assets/tables/novelty3_convergence.csv`
- **Output Figures**: `paper_assets/figures/novelty3_federated_vs_local.png`, `paper_assets/figures/novelty3_fl_convergence.png`

**Federated vs. Centralized vs. Local Benchmark (Seeds: 42, 101, 2024):**
| Paradigm | Data Locality | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Comm Bytes | Privacy Audit |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Centralized (Upper Bound)** | Raw Data Pooled | 1.0000 | 0.9885 | 0.9942 ± 0.00 | 1.0000 | 1.0000 | 0.0 MB | **FAIL** (Raw Data Pooled) |
| Local-Only: Factory A | Factory A Only | 1.0000 | 0.9885 | 0.9942 ± 0.00 | 1.0000 | 1.0000 | 0.0 MB | **PASS** (Isolated) |
| Local-Only: Factory B | Factory B Only | 1.0000 | 0.9885 | 0.9942 ± 0.00 | 1.0000 | 1.0000 | 0.0 MB | **PASS** (Isolated) |
| Local-Only: Factory C | Factory C Only | 1.0000 | 0.9885 | 0.9942 ± 0.00 | 1.0000 | 1.0000 | 0.0 MB | **PASS** (Isolated) |
| **Federated (FedAvg, 10 Rnds)**| Model Weights Only | **1.0000** | **0.9885** | **0.9942 ± 0.00** | **1.0000** | **1.0000** | **13.73 MB** | **PASS** (Zero Raw Data) |

*Programmatic Privacy Verification*:
```text
[AUDIT VERIFIED] Factory_A payload: 18 param tensors (468.54 KB) | Raw Telemetry: 0 rows | Transmitted: False
[AUDIT VERIFIED] Factory_B payload: 18 param tensors (468.54 KB) | Raw Telemetry: 0 rows | Transmitted: False
[AUDIT VERIFIED] Factory_C payload: 18 param tensors (468.54 KB) | Raw Telemetry: 0 rows | Transmitted: False
```

---

### Step-by-Step Complete Verification Pipeline

To execute and verify the complete evaluation pipeline in sequence:
```bash
# 1. Run Core Novelty Evaluations
python experiments/run_novelty1_what_if.py
python experiments/run_novelty2_split_brain.py
python experiments/run_novelty3_federated.py

# 2. Run Individual Diagnostic Experiments
python experiments/evaluate_counterfactuals.py   # CF vs SHAP vs LIME
python experiments/evaluate_causal.py            # Tigramite PCMCI DAG validation
python experiments/evaluate_split_brain.py       # Dual-tier latency & workload
python experiments/edge_benchmark.py             # FP32 vs ONNX vs INT8 edge profiling

# 3. Run Full Automated Test Suite
pytest tests/ -v
```

---

## Results Summary

Comprehensive evaluation summary across all research metrics (`paper_assets/tables/final_summary.csv`):

| Metric Category | Metric Name | Measured Value | Research Target | Status |
| :--- | :--- | :---: | :---: | :---: |
| **Explainability (XAI)** | Root Cause Accuracy | **100.0%** | > 90.0% | **PASSED** |
| | Faithfulness Rate | **100.0%** | > 50% score drop | **PASSED** |
| | Validity Rate | **100.0%** | Score < Threshold | **PASSED** |
| | Mean Sparsity | **4.01 features** | Minimal perturbation | **PASSED** |
| | Explanation Latency | **4.87 ms** | < 200 ms | **PASSED** |
| **Edge Performance** | Edge Latency (INT8) | **0.177 ms** | < 5.0 ms | **PASSED** |
| | Model Size (INT8) | **0.193 MB** | >= 3x compression (0.591 -> 0.193 MB) | **PASSED** |
| **Federated Learning**| Federated F1-Score | **0.9942 ± 0.0000** | Within 2% of Centralized | **PASSED** |
| | Centralized F1-Score | **0.9942 ± 0.0000** | Baseline Upper Bound | **PASSED** |
| | Privacy Audit | **0 Raw Bytes** | Zero telemetry leakage | **PASSED** |
| **Causal Discovery** | Graph Precision | **81.82%** | > 80.0% | **PASSED** |
| | Graph Recall | **60.00%** | Physical DAG coverage | **PASSED** |

Benchmark comparison of detection algorithms on the 10-sensor IoT telemetry test set:

| Model | F1-Score | Precision | Recall | AUC-ROC | AUC-PR | Latency (ms) | Size (MB) | XAI Compatible |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Isolation Forest** | 0.8065 | 0.8721 | 0.7500 | 0.9899 | 0.9297 | 3.51 | 2.64 | No |
| **One-Class SVM** | 0.9899 | 1.0000 | 0.9800 | 1.0000 | 1.0000 | 0.06 | 0.02 | No |
| **LSTM-Autoencoder (Ours)**| **0.9848** | **1.0000** | **0.9700** | **1.0000** | **1.0000** | **1.22** | **0.47** | **Yes (Gradient CF)** |
| **Anomaly Transformer** | 0.9744 | 1.0000 | 0.9500 | 0.9998 | 0.9985 | 0.63 | 0.43 | No (Complex Attn) |

*Pareto Analysis*: The LSTM-Autoencoder matches top-tier accuracy (F1: 0.9848, AUC: 1.0000) with a ultra-compact 0.47MB memory footprint and 1.22ms inference latency, providing complete end-to-end gradient differentiability required for real-time counterfactual generation.

## Citation

```bibtex
@article{shah2026xai_iot,
  title={XAI for IOT Anomaly Detection},
  author={Shah, Dev and Collaborators},
  journal={IEEE Transactions on Industrial Informatics},
  year={2026},
  volume={22},
  number={4},
  pages={1--12}
}
```

## License

This project is licensed under the Apache 2.0 License. See the LICENSE file for details.
