"""
Experimental Evaluation for Novelty 1: Actionable What-If Explanations.

Benchmarking across:
1. Proposed: Actionable Physics-Aware Causal Counterfactual
2. Baseline 1: Feature Ablation (Naively zeroing/median-clamping top reconstruction error feature)
3. Baseline 2: Unconstrained Gradient-Based Counterfactual (Adam optimization)

Metrics:
- Counterfactual Validity / Success Rate (%)
- Sparsity (number of perturbed features)
- Proximity (L1 and L2 distances)
- Actionability Rate (%)
- Explanation Generation Latency (ms)
- Anomaly Score Reduction (Absolute and %)
- Root Cause Attribution Accuracy (%)

Evaluates across multiple seeds with mean ± std.
Outputs:
- paper_assets/tables/novelty1_what_if_summary.csv
- paper_assets/tables/novelty1_per_anomaly_detail.csv
- paper_assets/figures/novelty1_validity_sparsity.png
- paper_assets/figures/novelty1_score_reduction.png
"""

import logging
import os
import sys
import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.detection.train import create_sliding_windows
from models.xai.actionable_what_if import ActionableWhatIfExplainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TRUE_CAUSES = {
    "A1_Fan_Failure": "fan_speed",
    "A2_Lubrication_Leak": "lubrication_flow",
    "A3_Heatwave": "ambient_temp",
    "A4_Sensor_Glitch": "vibration",
    "A5_Spindle_Overload": "spindle_speed",
}


def run_novelty1_experiments(
    config_path: str = "config/config.yaml",
    seeds: List[int] = [42, 123, 456],
    save_summary_csv: str = "paper_assets/tables/novelty1_what_if_summary.csv",
    save_detail_csv: str = "paper_assets/tables/novelty1_per_anomaly_detail.csv",
    save_fig1: str = "paper_assets/figures/novelty1_validity_sparsity.png",
    save_fig2: str = "paper_assets/figures/novelty1_score_reduction.png",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute complete multi-seed benchmarking suite for Novelty 1."""
    logger.info("=" * 70)
    logger.info("STARTING EXPERIMENT: NOVELTY 1 - ACTIONABLE WHAT-IF EXPLANATIONS")
    logger.info("=" * 70)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    explainer = ActionableWhatIfExplainer(config_path=config_path)
    detector = explainer.detector
    features = explainer.feature_names
    scaler = explainer.scaler

    df = pd.read_csv(config["data"]["synthetic_csv"])
    X_scaled = scaler.transform(df[features].values)
    y_raw = df["anomaly"].values
    anomaly_types = df["anomaly_type"].values

    window_size = config["preprocessing"]["window_size"]
    X_windows, y_windows = create_sliding_windows(X_scaled, y_raw, window_size=window_size)
    window_types = [anomaly_types[i + window_size - 1] for i in range(len(X_windows))]

    scores = detector.compute_anomaly_scores(X_windows)
    threshold = detector.threshold

    detected_anom_indices = [
        i for i in range(len(X_windows)) if scores[i] > threshold and window_types[i] in TRUE_CAUSES
    ]
    logger.info(f"Identified {len(detected_anom_indices)} detected failure instances for counterfactual generation.")

    # Storage for multi-seed aggregation
    seed_metrics = {
        "Proposed (Actionable What-If)": {"validity": [], "sparsity": [], "prox_l1": [], "prox_l2": [], "actionability": [], "latency": [], "reduction_pct": [], "accuracy": []},
        "Baseline 1 (Feature Ablation)": {"validity": [], "sparsity": [], "prox_l1": [], "prox_l2": [], "actionability": [], "latency": [], "reduction_pct": [], "accuracy": []},
        "Baseline 2 (Gradient Optimization)": {"validity": [], "sparsity": [], "prox_l1": [], "prox_l2": [], "actionability": [], "latency": [], "reduction_pct": [], "accuracy": []},
    }

    detailed_records = []

    for seed in seeds:
        np.random.seed(seed)
        logger.info(f"Running evaluation with random seed {seed}...")

        # Subsample for gradient baseline to maintain reasonable benchmarking time
        sample_indices = detected_anom_indices

        p_correct, b1_correct, b2_correct = 0, 0, 0
        p_val, b1_val, b2_val = [], [], []
        p_spar, b1_spar, b2_spar = [], [], []
        p_l1, b1_l1, b2_l1 = [], [], []
        p_l2, b1_l2, b2_l2 = [], [], []
        p_act, b1_act, b2_act = [], [], []
        p_lat, b1_lat, b2_lat = [], [], []
        p_red, b1_red, b2_red = [], [], []

        for idx in sample_indices:
            win = X_windows[idx]
            atype = window_types[idx]
            true_cause = TRUE_CAUSES[atype]

            # 1. Proposed Actionable What-If
            res_p = explainer.generate_counterfactual(win, threshold=threshold)
            p_val.append(res_p["validity"])
            p_spar.append(res_p["sparsity"])
            p_l1.append(res_p["proximity_l1"])
            p_l2.append(res_p["proximity_l2"])
            p_act.append(res_p["actionability"])
            p_lat.append(res_p["latency_ms"])
            p_red.append(res_p["score_reduction_pct"])
            if res_p["root_cause"] == true_cause:
                p_correct += 1

            # 2. Baseline 1: Feature Ablation
            res_b1 = explainer.generate_baseline_ablation(win, threshold=threshold)
            b1_val.append(res_b1["validity"])
            b1_spar.append(res_b1["sparsity"])
            b1_l1.append(res_b1["proximity_l1"])
            b1_l2.append(res_b1["proximity_l2"])
            b1_act.append(res_b1["actionability"])
            b1_lat.append(res_b1["latency_ms"])
            b1_red.append(res_b1["score_reduction_pct"])
            if res_b1["root_cause"] == true_cause:
                b1_correct += 1

            # 3. Baseline 2: Gradient Optimization
            # Run on subset of instances to keep runtime fast and responsive
            if idx % 5 == 0:
                res_b2 = explainer.generate_baseline_gradient(win, steps=80, threshold=threshold)
                b2_val.append(res_b2["validity"])
                b2_spar.append(res_b2["sparsity"])
                b2_l1.append(res_b2["proximity_l1"])
                b2_l2.append(res_b2["proximity_l2"])
                b2_act.append(res_b2["actionability"])
                b2_lat.append(res_b2["latency_ms"])
                b2_red.append(res_b2["score_reduction_pct"])
                if res_b2["root_cause"] == true_cause:
                    b2_correct += 1

            # Log first seed details to detailed records
            if seed == seeds[0]:
                detailed_records.append({
                    "Window_Index": idx,
                    "Scenario": atype,
                    "True_Root_Cause": true_cause,
                    "Proposed_Root_Cause": res_p["root_cause"],
                    "Proposed_Match": (res_p["root_cause"] == true_cause),
                    "Current_Reading": res_p["current_value"],
                    "Target_Reading": res_p["target_value"],
                    "Required_Change": res_p["required_change"],
                    "Direction": res_p["direction"],
                    "Unit": res_p["unit"],
                    "Orig_Score": res_p["original_score"],
                    "CF_Score": res_p["counterfactual_score"],
                    "Threshold": res_p["threshold"],
                    "Validity": res_p["validity"],
                    "Sparsity": res_p["sparsity"],
                    "Proximity_L1": res_p["proximity_l1"],
                    "Proximity_L2": res_p["proximity_l2"],
                    "Actionability_Pct": res_p["actionability"],
                    "Score_Reduction_Pct": res_p["score_reduction_pct"],
                    "Latency_ms": res_p["latency_ms"],
                    "Explanation_Narrative": res_p["explanation_narrative"],
                })

        n_inst = len(sample_indices)
        b2_n = len(b2_val)

        seed_metrics["Proposed (Actionable What-If)"]["validity"].append(np.mean(p_val) * 100.0)
        seed_metrics["Proposed (Actionable What-If)"]["sparsity"].append(np.mean(p_spar))
        seed_metrics["Proposed (Actionable What-If)"]["prox_l1"].append(np.mean(p_l1))
        seed_metrics["Proposed (Actionable What-If)"]["prox_l2"].append(np.mean(p_l2))
        seed_metrics["Proposed (Actionable What-If)"]["actionability"].append(np.mean(p_act))
        seed_metrics["Proposed (Actionable What-If)"]["latency"].append(np.mean(p_lat))
        seed_metrics["Proposed (Actionable What-If)"]["reduction_pct"].append(np.mean(p_red))
        seed_metrics["Proposed (Actionable What-If)"]["accuracy"].append((p_correct / n_inst) * 100.0)

        seed_metrics["Baseline 1 (Feature Ablation)"]["validity"].append(np.mean(b1_val) * 100.0)
        seed_metrics["Baseline 1 (Feature Ablation)"]["sparsity"].append(np.mean(b1_spar))
        seed_metrics["Baseline 1 (Feature Ablation)"]["prox_l1"].append(np.mean(b1_l1))
        seed_metrics["Baseline 1 (Feature Ablation)"]["prox_l2"].append(np.mean(b1_l2))
        seed_metrics["Baseline 1 (Feature Ablation)"]["actionability"].append(np.mean(b1_act))
        seed_metrics["Baseline 1 (Feature Ablation)"]["latency"].append(np.mean(b1_lat))
        seed_metrics["Baseline 1 (Feature Ablation)"]["reduction_pct"].append(np.mean(b1_red))
        seed_metrics["Baseline 1 (Feature Ablation)"]["accuracy"].append((b1_correct / n_inst) * 100.0)

        seed_metrics["Baseline 2 (Gradient Optimization)"]["validity"].append(np.mean(b2_val) * 100.0)
        seed_metrics["Baseline 2 (Gradient Optimization)"]["sparsity"].append(np.mean(b2_spar))
        seed_metrics["Baseline 2 (Gradient Optimization)"]["prox_l1"].append(np.mean(b2_l1))
        seed_metrics["Baseline 2 (Gradient Optimization)"]["prox_l2"].append(np.mean(b2_l2))
        seed_metrics["Baseline 2 (Gradient Optimization)"]["actionability"].append(np.mean(b2_act))
        seed_metrics["Baseline 2 (Gradient Optimization)"]["latency"].append(np.mean(b2_lat))
        seed_metrics["Baseline 2 (Gradient Optimization)"]["reduction_pct"].append(np.mean(b2_red))
        seed_metrics["Baseline 2 (Gradient Optimization)"]["accuracy"].append((b2_correct / b2_n) * 100.0)

    # Compile Summary Table with Mean ± Std
    summary_rows = []
    for method_name, m_dict in seed_metrics.items():
        summary_rows.append({
            "Method": method_name,
            "Root_Cause_Accuracy (%)": f"{np.mean(m_dict['accuracy']):.1f} ± {np.std(m_dict['accuracy']):.2f}",
            "Validity_Rate (%)": f"{np.mean(m_dict['validity']):.1f} ± {np.std(m_dict['validity']):.2f}",
            "Sparsity (Features)": f"{np.mean(m_dict['sparsity']):.2f} ± {np.std(m_dict['sparsity']):.2f}",
            "Proximity_L1": f"{np.mean(m_dict['prox_l1']):.4f} ± {np.std(m_dict['prox_l1']):.4f}",
            "Proximity_L2": f"{np.mean(m_dict['prox_l2']):.4f} ± {np.std(m_dict['prox_l2']):.4f}",
            "Actionability (%)": f"{np.mean(m_dict['actionability']):.1f} ± {np.std(m_dict['actionability']):.2f}",
            "Score_Reduction (%)": f"{np.mean(m_dict['reduction_pct']):.1f} ± {np.std(m_dict['reduction_pct']):.2f}",
            "Latency (ms)": f"{np.mean(m_dict['latency']):.2f} ± {np.std(m_dict['latency']):.2f}",
        })

    df_summary = pd.DataFrame(summary_rows)
    df_detail = pd.DataFrame(detailed_records)

    # Save Tables
    os.makedirs(os.path.dirname(save_summary_csv), exist_ok=True)
    os.makedirs(os.path.dirname(save_detail_csv), exist_ok=True)
    df_summary.to_csv(save_summary_csv, index=False)
    df_detail.to_csv(save_detail_csv, index=False)

    logger.info(f"Saved Novelty 1 summary table to {save_summary_csv}")
    logger.info(f"Saved Novelty 1 per-anomaly detail to {save_detail_csv}")

    # Plot 1: Validity and Sparsity Trade-off
    os.makedirs(os.path.dirname(save_fig1), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    methods = ["Proposed (Ours)", "Feature Ablation", "Gradient Opt"]
    val_means = [np.mean(seed_metrics[k]["validity"]) for k in seed_metrics]
    val_stds = [np.std(seed_metrics[k]["validity"]) for k in seed_metrics]
    spar_means = [np.mean(seed_metrics[k]["sparsity"]) for k in seed_metrics]
    spar_stds = [np.std(seed_metrics[k]["sparsity"]) for k in seed_metrics]
    colors = ["#27ae60", "#e74c3c", "#3498db"]

    ax1.bar(methods, val_means, yerr=val_stds, color=colors, capsize=5, edgecolor="black")
    ax1.set_ylabel("Validity / Success Rate (%)", fontweight="bold")
    ax1.set_title("Counterfactual Validity (Score < Threshold)", fontweight="bold")
    ax1.set_ylim(0, 115)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    for i, v in enumerate(val_means):
        ax1.text(i, v + 3, f"{v:.1f}%", ha="center", fontweight="bold")

    ax2.bar(methods, spar_means, yerr=spar_stds, color=colors, capsize=5, edgecolor="black")
    ax2.set_ylabel("Sparsity (Features Modified)", fontweight="bold")
    ax2.set_title("Explanation Sparsity (Lower is Better)", fontweight="bold")
    ax2.set_ylim(0, 11)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    for i, v in enumerate(spar_means):
        ax2.text(i, v + 0.3, f"{v:.2f}", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_fig1, dpi=300)
    plt.close()
    logger.info(f"Saved validity and sparsity figure to {save_fig1}")

    # Plot 2: Anomaly Score Reduction vs Actionability
    os.makedirs(os.path.dirname(save_fig2), exist_ok=True)
    fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))

    act_means = [np.mean(seed_metrics[k]["actionability"]) for k in seed_metrics]
    act_stds = [np.std(seed_metrics[k]["actionability"]) for k in seed_metrics]
    red_means = [np.mean(seed_metrics[k]["reduction_pct"]) for k in seed_metrics]
    red_stds = [np.std(seed_metrics[k]["reduction_pct"]) for k in seed_metrics]

    ax3.bar(methods, act_means, yerr=act_stds, color=colors, capsize=5, edgecolor="black")
    ax3.set_ylabel("Actionability Rate (%)", fontweight="bold")
    ax3.set_title("Actionability & Controllability Rate", fontweight="bold")
    ax3.set_ylim(0, 115)
    ax3.grid(axis="y", linestyle="--", alpha=0.5)
    for i, v in enumerate(act_means):
        ax3.text(i, v + 3, f"{v:.1f}%", ha="center", fontweight="bold")

    ax4.bar(methods, red_means, yerr=red_stds, color=colors, capsize=5, edgecolor="black")
    ax4.set_ylabel("Score Reduction (%)", fontweight="bold")
    ax4.set_title("Anomaly Score Drop Percentage", fontweight="bold")
    ax4.set_ylim(0, 115)
    ax4.grid(axis="y", linestyle="--", alpha=0.5)
    for i, v in enumerate(red_means):
        ax4.text(i, v + 3, f"{v:.1f}%", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_fig2, dpi=300)
    plt.close()
    logger.info(f"Saved score reduction and actionability figure to {save_fig2}")

    print("\n" + "=" * 75)
    print("NOVELTY 1: ACTIONABLE WHAT-IF EXPLANATIONS BENCHMARK SUMMARY (3 SEEDS)")
    print("=" * 75)
    print(df_summary.to_string(index=False))
    print("=" * 75 + "\n")

    return df_summary, df_detail


if __name__ == "__main__":
    run_novelty1_experiments()
