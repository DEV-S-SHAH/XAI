"""
Edge Deployment Benchmarking for XAI for IOT Anomaly Detection.
Empirically benchmarks PyTorch FP32, ONNX FP32, and ONNX INT8 models
across Storage Footprint (MB), Single-Sample Inference Latency (ms), and Process RAM Usage (MB).
Conducts 1000 inference iterations using psutil and time.perf_counter.
"""

import argparse
import logging
import os
import sys
import time
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import pandas as pd
import psutil
import seaborn as sns
import torch
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.detection.export_onnx import export_lstm_ae_to_onnx  # noqa: E402
from models.detection.lstm_autoencoder import LSTMAutoencoder  # noqa: E402
from models.detection.quantize import quantize_onnx_model  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})


def get_current_ram_mb() -> float:
    """Get resident memory usage of the current Python process in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def benchmark_pytorch_fp32(
    checkpoint_path: str, config: Dict, num_iterations: int = 1000
) -> Dict[str, float]:
    """Benchmark PyTorch FP32 model with mean and p95 latency."""
    logger.info("Benchmarking PyTorch FP32 model...")
    cfg_m = config["models"]["lstm_ae"]
    model = LSTMAutoencoder(
        input_dim=cfg_m["input_dim"],
        hidden_dim=cfg_m["hidden_dim"],
        seq_len=config["preprocessing"]["window_size"],
        num_layers=cfg_m["num_layers"],
        dropout=cfg_m["dropout"],
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu")["model_state_dict"])
    model.eval()

    sample = torch.randn(1, config["preprocessing"]["window_size"], cfg_m["input_dim"])
    for _ in range(50):
        with torch.no_grad():
            _ = model(sample)

    latencies = []
    with torch.no_grad():
        for _ in range(num_iterations):
            t_start = time.perf_counter()
            _ = model(sample)
            latencies.append((time.perf_counter() - t_start) * 1000.0)

    avg_ms = float(np.mean(latencies))
    p95_ms = float(np.percentile(latencies, 95))
    size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
    return {
        "Version": "PyTorch FP32",
        "Size (MB)": round(size_mb, 3),
        "Latency (ms)": round(avg_ms, 3),
        "Latency P95 (ms)": round(p95_ms, 3),
        "RAM (MB)": round(get_current_ram_mb(), 2),
    }


def benchmark_onnx(
    onnx_path: str, version_name: str, config: Dict, num_iterations: int = 1000
) -> Dict[str, float]:
    """Benchmark ONNX Runtime model (FP32 or INT8) with mean and p95 latency."""
    logger.info(f"Benchmarking {version_name} onnxruntime session...")
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inp_name = session.get_inputs()[0].name
    sample = np.random.randn(
        1, config["preprocessing"]["window_size"], config["models"]["lstm_ae"]["input_dim"]
    ).astype(np.float32)

    for _ in range(50):
        _ = session.run(None, {inp_name: sample})

    latencies = []
    for _ in range(num_iterations):
        t_start = time.perf_counter()
        _ = session.run(None, {inp_name: sample})
        latencies.append((time.perf_counter() - t_start) * 1000.0)

    avg_ms = float(np.mean(latencies))
    p95_ms = float(np.percentile(latencies, 95))
    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    return {
        "Version": version_name,
        "Size (MB)": round(size_mb, 3),
        "Latency (ms)": round(avg_ms, 3),
        "Latency P95 (ms)": round(p95_ms, 3),
        "RAM (MB)": round(get_current_ram_mb(), 2),
    }


def run_edge_benchmark(
    config_path: str = "config/config.yaml",
    iterations: int = 1000,
    save_csv_path: str = "paper_assets/tables/edge_benchmark.csv",
    save_plot_path: str = "paper_assets/figures/edge_latency.png",
) -> pd.DataFrame:
    """Run full edge benchmarking suite across PyTorch, ONNX FP32, and ONNX INT8."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    pt_path = config["models"]["lstm_ae"]["checkpoint_path"]
    onnx_fp32_path = config["edge"]["onnx_fp32"]
    onnx_int8_path = config["edge"]["onnx_int8"]

    # Ensure ONNX models exist
    if not os.path.exists(onnx_fp32_path):
        export_lstm_ae_to_onnx(pt_path, onnx_fp32_path, config_path)
    if not os.path.exists(onnx_int8_path):
        quantize_onnx_model(onnx_fp32_path, onnx_int8_path)

    results = [
        benchmark_pytorch_fp32(pt_path, config, num_iterations=iterations),
        benchmark_onnx(onnx_fp32_path, "ONNX FP32", config, num_iterations=iterations),
        benchmark_onnx(onnx_int8_path, "ONNX INT8", config, num_iterations=iterations),
    ]
    df = pd.DataFrame(results)
    sep = "=" * 70
    print(
        f"\n{sep}\nXAI for IOT Anomaly Detection: EDGE HARDWARE DEPLOYMENT BENCHMARK\n{sep}\n{df.to_string(index=False)}\n{sep}\n"
    )

    # Save CSV
    os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)
    df.to_csv(save_csv_path, index=False)
    logger.info(f"Saved benchmark results to {save_csv_path}")

    # Plot Visualizations
    os.makedirs(os.path.dirname(save_plot_path), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    palette = ["#3498db", "#2ecc71", "#e74c3c"]

    sns.barplot(
        data=df,
        x="Version",
        y="Size (MB)",
        ax=axes[0],
        hue="Version",
        palette=palette,
        legend=False,
    )
    axes[0].set_title("Storage Footprint (MB)", fontweight="bold")
    axes[0].set_ylabel("Megabytes")

    sns.barplot(
        data=df,
        x="Version",
        y="Latency (ms)",
        ax=axes[1],
        hue="Version",
        palette=palette,
        legend=False,
    )
    axes[1].set_title("Inference Latency (ms)", fontweight="bold")
    axes[1].set_ylabel("Milliseconds")

    sns.barplot(
        data=df, x="Version", y="RAM (MB)", ax=axes[2], hue="Version", palette=palette, legend=False
    )
    axes[2].set_title("Process Memory Footprint (MB)", fontweight="bold")
    axes[2].set_ylabel("Resident RAM (MB)")

    plt.tight_layout()
    plt.savefig(save_plot_path, dpi=300)
    plt.close()
    logger.info(f"Saved edge latency plot to {save_plot_path}")

    # Save dedicated edge_size.png
    fig_sz, ax_sz = plt.subplots(figsize=(7, 4.5))
    sns.barplot(
        data=df, x="Version", y="Size (MB)", ax=ax_sz, hue="Version", palette=palette, legend=False
    )
    ax_sz.set_title("Edge Storage Footprint by Model Runtime", fontweight="bold")
    ax_sz.set_ylabel("Model Size (MB)")
    plt.tight_layout()
    plt.savefig("paper_assets/figures/edge_size.png", dpi=300)
    plt.close()
    logger.info("Saved edge size plot to paper_assets/figures/edge_size.png")

    # Also render Split-Brain architecture diagram
    split_brain_path = "paper_assets/figures/architecture.png"
    render_split_brain_diagram(split_brain_path)

    return df


def render_split_brain_diagram(
    save_path: str = "paper_assets/figures/architecture.png",
) -> None:
    """Generate visual schematic of Split-Brain Edge/Cloud Architecture."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(10, 5))
    edge_box = FancyBboxPatch(
        (0.05, 0.25), 0.38, 0.6, boxstyle="round,pad=0.04", fc="#ecf0f1", ec="#2c3e50", lw=2
    )
    ax.add_patch(edge_box)
    ax.text(
        0.24,
        0.77,
        "EDGE TIER\n(Real-Time Ingestion)",
        ha="center",
        va="center",
        fontweight="bold",
        fontsize=11,
        color="#2c3e50",
    )
    edge_desc = "- Sensor Stream (1 min/sample)\n- Sliding Window Buffer (60 min)\n- ONNX INT8 Engine\n- Latency: < 2.0 ms\n- RAM: < 50 MB"
    ax.text(0.24, 0.55, edge_desc, ha="center", va="center", fontsize=9, family="monospace")
    ax.text(
        0.24,
        0.32,
        "POST /detect",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.2", fc="#27ae60", ec="none"),
        color="white",
        fontweight="bold",
    )
    ax.annotate(
        "",
        xy=(0.57, 0.55),
        xytext=(0.43, 0.55),
        arrowprops=dict(arrowstyle="->", lw=3, color="#e74c3c"),
    )
    ax.text(
        0.50,
        0.60,
        "Anomaly Trigger\n(Score > Threshold)",
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        color="#c0392b",
    )
    cloud_box = FancyBboxPatch(
        (0.57, 0.25), 0.38, 0.6, boxstyle="round,pad=0.04", fc="#e8f8f5", ec="#16a085", lw=2
    )
    ax.add_patch(cloud_box)
    ax.text(
        0.76,
        0.77,
        "EXPLANATION TIER\n(Causal Edge Server)",
        ha="center",
        va="center",
        fontweight="bold",
        fontsize=11,
        color="#16a085",
    )
    cloud_desc = "- Counterfactual Optimization\n- Causal DAG Validation\n- Action Formatter\n- Latency: ~100 ms"
    ax.text(0.76, 0.55, cloud_desc, ha="center", va="center", fontsize=9, family="monospace")
    ax.text(
        0.76,
        0.32,
        "POST /explain",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.2", fc="#2980b9", ec="none"),
        color="white",
        fontweight="bold",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0.1, 0.95)
    ax.axis("off")
    plt.title("XAI for IOT Anomaly Detection: Split-Brain IoT Edge Architecture", fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Edge Benchmark.")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()

    run_edge_benchmark(config_path=args.config, iterations=args.iterations)
