"""
Experimental Evaluation for Novelty 3: Privacy-Preserving Federated Training.

Benchmarks:
1. Centralized Training (Pooled data)
2. Local-Only Training (Isolated factories A, B, and C without federation)
3. Federated Training (FedAvg across 10 communication rounds)

Evaluates:
- Non-IID client distributions across 3 factory sites.
- Metrics: Precision, Recall, F1, ROC-AUC, PR-AUC.
- Convergence over 10 rounds (client loss, global F1).
- Communication volume in MB.
- Training time in seconds.
- Programmatic privacy verification.

Outputs:
- paper_assets/tables/novelty3_federated_comparison.csv
- paper_assets/tables/novelty3_convergence.csv
- paper_assets/tables/novelty3_privacy_audit.csv
- paper_assets/figures/novelty3_fl_convergence.png
- paper_assets/figures/novelty3_federated_vs_local.png
- paper_assets/tables/fl_convergence.csv
- paper_assets/figures/fl_convergence.png
"""

import logging
import os
import sys
import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.federated.federated_experiment import run_federated_vs_baselines

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_novelty3_experiments(
    config_path: str = "config/config.yaml",
    seeds: List[int] = [42, 101, 2024],
    save_comp_csv: str = "paper_assets/tables/novelty3_federated_comparison.csv",
    save_conv_csv: str = "paper_assets/tables/novelty3_convergence.csv",
    save_fig_conv: str = "paper_assets/figures/novelty3_fl_convergence.png",
    save_fig_bar: str = "paper_assets/figures/novelty3_federated_vs_local.png",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute complete multi-seed federated benchmarking suite."""
    logger.info("=" * 70)
    logger.info("STARTING EXPERIMENT: NOVELTY 3 - PRIVACY-PRESERVING FEDERATED TRAINING")
    logger.info("=" * 70)

    all_seed_comparisons = []
    last_df_conv = None

    for seed in seeds:
        logger.info(f"Running Federated experiment with seed {seed}...")
        res_dict, df_comp, df_conv = run_federated_vs_baselines(
            config_path=config_path,
            num_rounds=10,
            local_epochs_per_round=2,
            seed=seed,
        )
        df_comp["Seed"] = seed
        all_seed_comparisons.append(df_comp)
        last_df_conv = df_conv

    # Aggregate across seeds
    combined_df = pd.concat(all_seed_comparisons, ignore_index=True)
    grouped = combined_df.groupby("Paradigm")

    summary_rows = []
    paradigms = combined_df["Paradigm"].unique()

    for p in paradigms:
        sub = combined_df[combined_df["Paradigm"] == p]
        summary_rows.append({
            "Paradigm": p,
            "Data_Locality": sub["Data_Locality"].iloc[0],
            "Precision": f"{sub['Precision'].mean():.4f} ± {sub['Precision'].std():.4f}",
            "Recall": f"{sub['Recall'].mean():.4f} ± {sub['Recall'].std():.4f}",
            "F1_Score": f"{sub['F1_Score'].mean():.4f} ± {sub['F1_Score'].std():.4f}",
            "ROC_AUC": f"{sub['ROC_AUC'].mean():.4f} ± {sub['ROC_AUC'].std():.4f}",
            "PR_AUC": f"{sub['PR_AUC'].mean():.4f} ± {sub['PR_AUC'].std():.4f}",
            "Comm_Bytes_MB": sub["Comm_Bytes_MB"].mean(),
            "Training_Time_s": round(sub["Training_Time_s"].mean(), 2),
            "Privacy_Preserved": sub["Privacy_Preserved"].iloc[0],
        })

    df_final_comp = pd.DataFrame(summary_rows)

    # Save Tables
    os.makedirs(os.path.dirname(save_comp_csv), exist_ok=True)
    os.makedirs(os.path.dirname(save_conv_csv), exist_ok=True)
    df_final_comp.to_csv(save_comp_csv, index=False)
    last_df_conv.to_csv(save_conv_csv, index=False)
    # Also save to canonical paths
    last_df_conv.to_csv("paper_assets/tables/fl_convergence.csv", index=False)

    logger.info(f"Saved Novelty 3 comparison table to {save_comp_csv}")
    logger.info(f"Saved Novelty 3 convergence table to {save_conv_csv}")

    # ==========================================
    # PLOTS
    # ==========================================
    # Plot 1: Federated Learning Convergence over 10 Rounds
    os.makedirs(os.path.dirname(save_fig_conv), exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax1.plot(last_df_conv["Round"], last_df_conv["Factory_A_Loss"], "o--", label="Factory A (Loss)", color="#3498db")
    ax1.plot(last_df_conv["Round"], last_df_conv["Factory_B_Loss"], "s--", label="Factory B (Loss)", color="#2ecc71")
    ax1.plot(last_df_conv["Round"], last_df_conv["Factory_C_Loss"], "^--", label="Factory C (Loss)", color="#f39c12")
    ax1.plot(last_df_conv["Round"], last_df_conv["Avg_Client_Loss"], "k-", lw=2.2, label="FedAvg Aggregated Loss")

    ax1.set_xlabel("Federated Communication Round", fontweight="bold")
    ax1.set_ylabel("Local Reconstruction Loss (MSE)", fontweight="bold")
    ax1.set_xticks(range(1, 11))
    ax1.legend(loc="center right")
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2 = ax1.twinx()
    ax2.plot(last_df_conv["Round"], last_df_conv["Global_F1"], "r-o", lw=2.5, label="Global Model F1-Score")
    ax2.set_ylabel("Global Model F1-Score", color="red", fontweight="bold")
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.set_ylim(0.70, 1.05)

    plt.title("Federated Learning (FedAvg) Convergence across 3 Decentralized IoT Sites", fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(save_fig_conv, dpi=300)
    plt.savefig("paper_assets/figures/fl_convergence.png", dpi=300)
    plt.close()
    logger.info(f"Saved FL convergence figure to {save_fig_conv}")

    # Plot 2: Federated vs Local vs Centralized Bar Chart
    os.makedirs(os.path.dirname(save_fig_bar), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))

    labels = ["Centralized\n(Pooled)", "Local A\n(Isolated)", "Local B\n(Isolated)", "Local C\n(Isolated)", "Federated\n(FedAvg 10R)"]
    f1_means = [
        float(df_final_comp.loc[df_final_comp["Paradigm"] == p, "F1_Score"].iloc[0].split(" ± ")[0])
        for p in df_final_comp["Paradigm"]
    ]
    auc_means = [
        float(df_final_comp.loc[df_final_comp["Paradigm"] == p, "ROC_AUC"].iloc[0].split(" ± ")[0])
        for p in df_final_comp["Paradigm"]
    ]

    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width/2, f1_means, width, label="F1-Score", color="#2980b9", edgecolor="black")
    ax.bar(x + width/2, auc_means, width, label="ROC-AUC", color="#27ae60", edgecolor="black")

    ax.set_ylabel("Score (0.0 to 1.0)", fontweight="bold")
    ax.set_title("Detection Performance: Centralized vs. Isolated Local vs. Federated", fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylim(0.4, 1.15)
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for i in range(len(x)):
        ax.text(x[i] - width/2, f1_means[i] + 0.02, f"{f1_means[i]:.2f}", ha="center", fontweight="bold", fontsize=9)
        ax.text(x[i] + width/2, auc_means[i] + 0.02, f"{auc_means[i]:.2f}", ha="center", fontweight="bold", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_fig_bar, dpi=300)
    plt.savefig("paper_assets/figures/fl_vs_centralized.png", dpi=300)
    plt.close()
    logger.info(f"Saved federated vs local comparison figure to {save_fig_bar}")

    print("\n" + "=" * 85)
    print("NOVELTY 3: FEDERATED LEARNING EXPERIMENTAL COMPARISON (3 SEEDS)")
    print("=" * 85)
    print(df_final_comp.to_string(index=False))
    print("=" * 85 + "\n")

    return df_final_comp, last_df_conv


if __name__ == "__main__":
    run_novelty3_experiments()
