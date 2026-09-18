"""
Experimental Evaluation for Novelty 2: Split-Brain Edge Architecture.

Benchmarks:
1. Baseline Centralized Architecture (Detection + Heavy XAI for every observation)
2. Proposed Split-Brain Architecture (Lightweight Edge ONNX INT8 + Conditional Server XAI)

Evaluates:
- Measured Metrics: Detection Latency, XAI Latency, End-to-End Latency, CPU %, RAM (MB), Model Size (MB),
  Network Transmission (KB), Executed vs Avoided XAI counts.
- Modeled Computational & Energy Proxies: GFLOPs proxy, Joule energy proxy.
- Parametric Anomaly Rate Sweep: 1%, 5%, 10%, 20%, 50%.

Outputs:
- paper_assets/tables/novelty2_split_brain_benchmark.csv
- paper_assets/tables/novelty2_anomaly_rate_sweep.csv
- paper_assets/figures/novelty2_latency_breakdown.png
- paper_assets/figures/novelty2_bandwidth_energy_tradeoff.png
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
from models.detection.split_brain_engine import SplitBrainEdgeEngine
from models.detection.train import create_sliding_windows

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_novelty2_experiments(
    config_path: str = "config/config.yaml",
    save_bench_csv: str = "paper_assets/tables/novelty2_split_brain_benchmark.csv",
    save_sweep_csv: str = "paper_assets/tables/novelty2_anomaly_rate_sweep.csv",
    save_fig_latency: str = "paper_assets/figures/novelty2_latency_breakdown.png",
    save_fig_tradeoff: str = "paper_assets/figures/novelty2_bandwidth_energy_tradeoff.png",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run full experimental suite for Split-Brain Edge Architecture."""
    logger.info("=" * 70)
    logger.info("STARTING EXPERIMENT: NOVELTY 2 - SPLIT-BRAIN EDGE ARCHITECTURE")
    logger.info("=" * 70)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    engine = SplitBrainEdgeEngine(config_path=config_path)

    # Load dataset
    df = pd.read_csv(config["data"]["synthetic_csv"])
    features = config["data"]["features"]
    scaler = engine.server_explainer.scaler

    X_scaled = scaler.transform(df[features].values)
    y_raw = df["anomaly"].values
    window_size = config["preprocessing"]["window_size"]
    X_windows, y_windows = create_sliding_windows(X_scaled, y_raw, window_size=window_size)

    # 1. Primary Empirical Benchmark (Full Dataset, ~4.0% anomaly prevalence)
    # Use 300 observations to keep execution responsive and statistically sound
    test_subsample = X_windows[:300]
    test_y_subsample = y_windows[:300]

    bench_res = engine.run_benchmark_comparison(test_subsample, test_y_subsample)

    b = bench_res["baseline"]
    p = bench_res["proposed"]
    s = bench_res["savings"]

    bench_rows = [
        # Measured Metrics
        {"Category": "Measured Performance", "Metric": "Detection Latency Mean (ms)", "Baseline (Centralized)": round(b["detection_latency_mean_ms"], 3), "Proposed (Split-Brain)": round(p["detection_latency_mean_ms"], 3), "Advantage / Delta": f"{b['detection_latency_mean_ms']/p['detection_latency_mean_ms']:.1f}x faster"},
        {"Category": "Measured Performance", "Metric": "XAI Latency Mean (ms)", "Baseline (Centralized)": round(b["xai_latency_mean_ms"], 3), "Proposed (Split-Brain)": round(p["xai_latency_mean_ms"], 3), "Advantage / Delta": "Unchanged (Server tier)"},
        {"Category": "Measured Performance", "Metric": "End-to-End Latency Mean (ms)", "Baseline (Centralized)": round(b["e2e_latency_mean_ms"], 3), "Proposed (Split-Brain)": round(p["e2e_latency_mean_ms"], 3), "Advantage / Delta": f"{s['latency_speedup_x']:.1f}x speedup"},
        {"Category": "Measured Performance", "Metric": "End-to-End Latency P95 (ms)", "Baseline (Centralized)": round(b["e2e_latency_p95_ms"], 3), "Proposed (Split-Brain)": round(p["e2e_latency_p95_ms"], 3), "Advantage / Delta": f"{b['e2e_latency_p95_ms']/p['e2e_latency_p95_ms']:.1f}x speedup"},
        {"Category": "Measured Hardware", "Metric": "Edge Model Size (MB)", "Baseline (Centralized)": round(b["model_size_mb"], 3), "Proposed (Split-Brain)": round(p["model_size_mb"], 3), "Advantage / Delta": f"{s['model_size_compression_x']:.2f}x compression"},
        {"Category": "Measured Hardware", "Metric": "Resident RAM (MB)", "Baseline (Centralized)": round(b["ram_mb"], 2), "Proposed (Split-Brain)": round(p["ram_mb"], 2), "Advantage / Delta": "Shared edge runtime"},
        {"Category": "Measured Network", "Metric": "Data Transmitted (KB)", "Baseline (Centralized)": round(b["data_transmitted_kb"], 1), "Proposed (Split-Brain)": round(p["data_transmitted_kb"], 1), "Advantage / Delta": f"{s['bandwidth_reduction_pct']:.1f}% bandwidth reduction"},
        {"Category": "Measured Execution", "Metric": "Total Ingested Observations", "Baseline (Centralized)": b["xai_executions"], "Proposed (Split-Brain)": bench_res["num_observations"], "Advantage / Delta": "100% Ingestion coverage"},
        {"Category": "Measured Execution", "Metric": "Server XAI Invocations", "Baseline (Centralized)": b["xai_executions"], "Proposed (Split-Brain)": p["xai_executions"], "Advantage / Delta": f"{p['avoided_xai_pct']:.1f}% calls avoided"},
        {"Category": "Measured Execution", "Metric": "Avoided XAI Executions", "Baseline (Centralized)": 0, "Proposed (Split-Brain)": p["avoided_xai_executions"], "Advantage / Delta": f"{p['avoided_xai_pct']:.1f}% of workload avoided"},

        # Modeled Computational & Energy Proxies (Explicitly labeled)
        {"Category": "Modeled Proxy", "Metric": "Computation Proxy (GFLOPs) [ESTIMATED]", "Baseline (Centralized)": round(b["flops_proxy_gflops"], 2), "Proposed (Split-Brain)": round(p["flops_proxy_gflops"], 2), "Advantage / Delta": f"{(1 - p['flops_proxy_gflops']/b['flops_proxy_gflops'])*100:.1f}% FLOPs reduction"},
        {"Category": "Modeled Proxy", "Metric": "Energy Consumption Proxy (Joules) [ESTIMATED]", "Baseline (Centralized)": round(b["energy_proxy_joules"], 2), "Proposed (Split-Brain)": round(p["energy_proxy_joules"], 2), "Advantage / Delta": f"{s['energy_reduction_pct']:.1f}% energy proxy reduction"},
    ]

    df_bench = pd.DataFrame(bench_rows)
    os.makedirs(os.path.dirname(save_bench_csv), exist_ok=True)
    df_bench.to_csv(save_bench_csv, index=False)
    logger.info(f"Saved Novelty 2 primary benchmark table to {save_bench_csv}")

    # ==========================================
    # 2. PARAMETRIC SWEEP OVER ANOMALY RATES
    # Anomaly Rates: 1%, 5%, 10%, 20%, 50%
    # ==========================================
    logger.info("Executing parametric sweep over anomaly rates: [1%, 5%, 10%, 20%, 50%]...")
    anomaly_rates = [0.01, 0.05, 0.10, 0.20, 0.50]
    sweep_rows = []

    # Separate normal windows from true anomaly windows
    norm_indices = [i for i in range(len(X_windows)) if y_windows[i] == 0]
    anom_indices = [i for i in range(len(X_windows)) if y_windows[i] == 1]

    N_SIM = 200

    for rate in anomaly_rates:
        n_anom = int(N_SIM * rate)
        n_norm = N_SIM - n_anom

        selected_norm = np.random.choice(norm_indices, size=n_norm, replace=True)
        selected_anom = np.random.choice(anom_indices, size=n_anom, replace=True)
        sim_indices = np.concatenate([selected_norm, selected_anom])
        np.random.shuffle(sim_indices)

        sim_windows = X_windows[sim_indices]
        sim_flags = y_windows[sim_indices]

        # Evaluate Split-Brain vs Baseline on this specific anomaly rate distribution
        res = engine.run_benchmark_comparison(sim_windows, sim_flags)
        b_sim = res["baseline"]
        p_sim = res["proposed"]
        s_sim = res["savings"]

        sweep_rows.append({
            "Anomaly_Rate_Pct": f"{int(rate * 100)}%",
            "Total_Observations": N_SIM,
            "Anomalies_Count": p_sim["xai_executions"],
            "Baseline_E2E_Latency_ms": round(b_sim["e2e_latency_mean_ms"], 2),
            "Proposed_E2E_Latency_ms": round(p_sim["e2e_latency_mean_ms"], 2),
            "Latency_Speedup": f"{s_sim['latency_speedup_x']:.2f}x",
            "Baseline_Data_TX_KB": round(b_sim["data_transmitted_kb"], 1),
            "Proposed_Data_TX_KB": round(p_sim["data_transmitted_kb"], 1),
            "Bandwidth_Savings_Pct": round(s_sim["bandwidth_reduction_pct"], 1),
            "Avoided_XAI_Calls_Pct": round(p_sim["avoided_xai_pct"], 1),
            "Baseline_Energy_Joules_Est": round(b_sim["energy_proxy_joules"], 2),
            "Proposed_Energy_Joules_Est": round(p_sim["energy_proxy_joules"], 2),
            "Energy_Savings_Pct_Est": round(s_sim["energy_reduction_pct"], 1),
        })

    df_sweep = pd.DataFrame(sweep_rows)
    os.makedirs(os.path.dirname(save_sweep_csv), exist_ok=True)
    df_sweep.to_csv(save_sweep_csv, index=False)
    logger.info(f"Saved Novelty 2 anomaly rate sweep table to {save_sweep_csv}")

    # ==========================================
    # 3. PLOTS & PUBLICATION VISUALIZATIONS
    # ==========================================
    # Plot 1: End-to-End Latency Breakdown
    os.makedirs(os.path.dirname(save_fig_latency), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    cats = ["Detection Only (Normal)", "Detection + XAI (Anomaly)", "Average Stream End-to-End"]
    base_lats = [b["e2e_latency_mean_ms"], b["e2e_latency_mean_ms"], b["e2e_latency_mean_ms"]]
    prop_lats = [p["detection_latency_mean_ms"], p["detection_latency_mean_ms"] + p["xai_latency_mean_ms"], p["e2e_latency_mean_ms"]]

    x = np.arange(len(cats))
    width = 0.35

    ax1.bar(x - width/2, base_lats, width, label="Baseline Centralized", color="#e74c3c", edgecolor="black")
    ax1.bar(x + width/2, prop_lats, width, label="Proposed Split-Brain", color="#27ae60", edgecolor="black")
    ax1.set_ylabel("Processing Latency (ms)", fontweight="bold")
    ax1.set_title("Single-Observation Processing Latency Breakdown", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(cats, fontsize=9)
    ax1.legend()
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Subplot 2: Latency Speedup across Anomaly Rates
    rates_num = [1, 5, 10, 20, 50]
    speedups = [float(row["Latency_Speedup"].replace("x", "")) for row in sweep_rows]
    ax2.plot(rates_num, speedups, "o-", color="#2980b9", lw=2.5, markersize=8)
    ax2.set_xlabel("Anomaly Prevalence Rate (%)", fontweight="bold")
    ax2.set_ylabel("Speedup Factor (Baseline / Proposed)", fontweight="bold")
    ax2.set_title("Effective System Speedup vs. Anomaly Prevalence", fontweight="bold")
    ax2.set_xticks(rates_num)
    ax2.grid(True, linestyle="--", alpha=0.5)
    for r, sp in zip(rates_num, speedups):
        ax2.text(r, sp + 0.5, f"{sp:.1f}x", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_fig_latency, dpi=300)
    plt.close()
    logger.info(f"Saved latency breakdown figure to {save_fig_latency}")

    # Plot 2: Bandwidth & Energy Trade-off
    os.makedirs(os.path.dirname(save_fig_tradeoff), exist_ok=True)
    fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))

    bw_savings = [row["Bandwidth_Savings_Pct"] for row in sweep_rows]
    en_savings = [row["Energy_Savings_Pct_Est"] for row in sweep_rows]

    ax3.plot(rates_num, bw_savings, "s-", color="#8e44ad", lw=2.5, markersize=8, label="Bandwidth Reduction")
    ax3.set_xlabel("Anomaly Prevalence Rate (%)", fontweight="bold")
    ax3.set_ylabel("Data Transmission Savings (%)", fontweight="bold")
    ax3.set_title("Network Bandwidth Savings across Anomaly Rates", fontweight="bold")
    ax3.set_xticks(rates_num)
    ax3.set_ylim(40, 102)
    ax3.grid(True, linestyle="--", alpha=0.5)
    for r, s_val in zip(rates_num, bw_savings):
        ax3.text(r, s_val + 1.5, f"{s_val:.1f}%", ha="center", fontweight="bold")

    ax4.plot(rates_num, en_savings, "^-", color="#d35400", lw=2.5, markersize=8, label="Energy Savings [Estimated Proxy]")
    ax4.set_xlabel("Anomaly Prevalence Rate (%)", fontweight="bold")
    ax4.set_ylabel("Energy Proxy Reduction (%)", fontweight="bold")
    ax4.set_title("Modeled Energy Consumption Savings Proxy", fontweight="bold")
    ax4.set_xticks(rates_num)
    ax4.set_ylim(40, 102)
    ax4.grid(True, linestyle="--", alpha=0.5)
    for r, e_val in zip(rates_num, en_savings):
        ax4.text(r, e_val + 1.5, f"{e_val:.1f}%", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_fig_tradeoff, dpi=300)
    plt.close()
    logger.info(f"Saved bandwidth and energy tradeoff figure to {save_fig_tradeoff}")

    print("\n" + "=" * 80)
    print("NOVELTY 2: SPLIT-BRAIN EDGE ARCHITECTURE BENCHMARK SUMMARY")
    print("=" * 80)
    print(df_bench.to_string(index=False))
    print("\nPARAMETRIC ANOMALY RATE SWEEP:")
    print(df_sweep.to_string(index=False))
    print("=" * 80 + "\n")

    return df_bench, df_sweep


if __name__ == "__main__":
    run_novelty2_experiments()
