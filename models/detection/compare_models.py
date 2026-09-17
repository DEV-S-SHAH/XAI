"""Model Comparison and Evaluation for XAI for IOT Anomaly Detection across 4 detectors, generating tables and curves."""

import argparse
import logging
import os
import sys
import time
from typing import Any, Dict, Tuple

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import yaml  # noqa: E402

from models.detection.anomaly_transformer_model import AnomalyTransformerDetector  # noqa: E402
from models.detection.isolation_forest_model import IsolationForestDetector  # noqa: E402
from models.detection.lstm_autoencoder import LSTMAutoencoderDetector  # noqa: E402
from models.detection.one_class_svm_model import OneClassSVMDetector  # noqa: E402
from models.detection.train import prepare_dataset  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Apply publication style
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})


def measure_inference_latency(model: Any, X_sample: np.ndarray, num_runs: int = 100) -> float:
    """Measure single-sample inference latency in milliseconds."""
    sample = X_sample[:1]
    for _ in range(10):
        _ = model.compute_anomaly_scores(sample)
    t0 = time.perf_counter()
    for _ in range(num_runs):
        _ = model.compute_anomaly_scores(sample)
    return float(((time.perf_counter() - t0) / num_runs) * 1000.0)


def get_model_size_mb(checkpoint_path: str) -> float:
    """Calculate file size of model checkpoint in Megabytes."""
    return (
        round(os.path.getsize(checkpoint_path) / (1024 * 1024), 3)
        if os.path.exists(checkpoint_path)
        else 0.0
    )


def compute_classification_metrics(
    y_true: np.ndarray, scores: np.ndarray, threshold: float
) -> Dict[str, float]:
    """Compute standard classification metrics from anomaly scores and threshold."""
    preds = (scores > threshold).astype(int)
    try:
        roc_auc = float(roc_auc_score(y_true, scores))
    except Exception:
        roc_auc = 0.5
    return {
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "roc_auc": roc_auc,
    }


def evaluate_all_models(
    config_path: str = "config/config.yaml",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load trained models, evaluate on test set, and produce comparative metrics."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    os.makedirs(config["paths"].get("paper_tables", "paper_assets/tables"), exist_ok=True)
    os.makedirs(config["paths"].get("paper_figures", "paper_assets/figures"), exist_ok=True)

    data_pkg = prepare_dataset(
        csv_path=config["data"]["synthetic_csv"],
        window_size=config["preprocessing"]["window_size"],
        train_ratio=config["preprocessing"]["train_ratio"],
        val_ratio=config["preprocessing"]["val_ratio"],
        test_ratio=config["preprocessing"]["test_ratio"],
    )

    X_test = data_pkg["X_test"]
    y_test = data_pkg["y_test"]

    logger.info(f"Evaluating models on test set: {len(X_test)} samples, {y_test.sum()} anomalies.")

    m_cfg = config["models"]
    models = {
        "Isolation Forest": {
            "detector": IsolationForestDetector().load(
                m_cfg["isolation_forest"]["checkpoint_path"]
            ),
            "xai_compat": "No",
            "checkpoint": m_cfg["isolation_forest"]["checkpoint_path"],
        },
        "One-Class SVM": {
            "detector": OneClassSVMDetector().load(m_cfg["one_class_svm"]["checkpoint_path"]),
            "xai_compat": "No",
            "checkpoint": m_cfg["one_class_svm"]["checkpoint_path"],
        },
        "LSTM-Autoencoder": {
            "detector": LSTMAutoencoderDetector(
                input_dim=m_cfg["lstm_ae"]["input_dim"],
                hidden_dim=m_cfg["lstm_ae"]["hidden_dim"],
                seq_len=config["preprocessing"]["window_size"],
            ).load(m_cfg["lstm_ae"]["checkpoint_path"]),
            "xai_compat": "Yes",
            "checkpoint": m_cfg["lstm_ae"]["checkpoint_path"],
        },
        "Anomaly Transformer": {
            "detector": AnomalyTransformerDetector(
                input_dim=m_cfg["anomaly_transformer"]["input_dim"],
                d_model=m_cfg["anomaly_transformer"]["d_model"],
                n_heads=m_cfg["anomaly_transformer"]["n_heads"],
                seq_len=config["preprocessing"]["window_size"],
            ).load(m_cfg["anomaly_transformer"]["checkpoint_path"]),
            "xai_compat": "No (Complex Attn)",
            "checkpoint": m_cfg["anomaly_transformer"]["checkpoint_path"],
        },
    }

    results = []
    curves_data = {}

    for name, item in models.items():
        detector = item["detector"]
        scores = detector.compute_anomaly_scores(X_test)
        preds = detector.predict(X_test)

        f1 = f1_score(y_test, preds, zero_division=0)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)

        # ROC & AUC
        fpr, tpr, _ = roc_curve(y_test, scores)
        roc_auc = roc_auc_score(y_test, scores)

        # PR Curve & AUC-PR
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, scores)
        pr_auc = auc(recall_vals, precision_vals)

        latency = measure_inference_latency(detector, X_test, num_runs=50)
        size_mb = get_model_size_mb(item["checkpoint"])

        results.append(
            {
                "Model": name,
                "F1-Score": round(float(f1), 4),
                "Precision": round(float(prec), 4),
                "Recall": round(float(rec), 4),
                "AUC-ROC": round(float(roc_auc), 4),
                "AUC-PR": round(float(pr_auc), 4),
                "Latency (ms)": round(latency, 2),
                "Size (MB)": round(size_mb, 2),
                "XAI Compatible": item["xai_compat"],
            }
        )
        curves_data[name] = {
            "fpr": fpr,
            "tpr": tpr,
            "roc_auc": roc_auc,
            "precision": precision_vals,
            "recall": recall_vals,
            "pr_auc": pr_auc,
            "preds": preds,
            "scores": scores,
        }

    df_results = pd.DataFrame(results)
    print(
        "\n"
        + "=" * 80
        + "\nXAI for IOT Anomaly Detection: BENCHMARK EVALUATION OF ANOMALY DETECTION ARCHITECTURES\n"
        + "=" * 80
    )
    print(df_results.to_string(index=False) + "\n" + "=" * 80 + "\n")

    csv_out = os.path.join(
        config["paths"].get("paper_tables", "paper_assets/tables"),
        "model_comparison.csv",
    )
    df_results.to_csv(csv_out, index=False)
    logger.info(f"Saved comparison table to {csv_out}")

    # Generate Visualizations
    fig_dir = config["paths"].get("paper_figures", "paper_assets/figures")

    save_comparison_plots(df_results, curves_data, y_test, fig_dir)
    return df_results, curves_data


def save_comparison_plots(
    df_results: pd.DataFrame, curves_data: Dict[str, Any], y_test: np.ndarray, fig_dir: str
) -> None:
    """Save publication-quality benchmark curves and confusion matrix."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    palette = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    metrics_spec = [
        ("F1-Score", "F1-Score Comparison", (0, 1.05)),
        ("Latency (ms)", "Inference Latency (ms)", None),
        ("Size (MB)", "Model Footprint (MB)", None),
    ]
    for idx, (col, title, ylim) in enumerate(metrics_spec):
        sns.barplot(data=df_results, x="Model", y=col, ax=axes[idx], palette=palette)
        axes[idx].set_title(title, fontweight="bold")
        if ylim:
            axes[idx].set_ylim(ylim)
        axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "f1_comparison.png"), dpi=300)
    plt.close()

    colors = {
        "Isolation Forest": "#1f77b4",
        "One-Class SVM": "#ff7f0e",
        "LSTM-Autoencoder": "#2ca02c",
        "Anomaly Transformer": "#d62728",
    }
    plt.figure(figsize=(8, 6))
    for name, cdata in curves_data.items():
        plt.plot(
            cdata["fpr"],
            cdata["tpr"],
            label=f"{name} (AUC = {cdata['roc_auc']:.3f})",
            lw=2,
            color=colors[name],
        )
    plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curves", fontweight="bold")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(fig_dir, "roc_curves.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    for name, cdata in curves_data.items():
        plt.plot(
            cdata["recall"],
            cdata["precision"],
            label=f"{name} (AUC-PR = {cdata['pr_auc']:.3f})",
            lw=2,
            color=colors[name],
        )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall (PR) Curves for Imbalanced IoT Detection", fontweight="bold")
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(fig_dir, "pr_curves.png"), dpi=300)
    plt.close()

    cm = confusion_matrix(y_test, curves_data["LSTM-Autoencoder"]["preds"])
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Anomaly"],
        yticklabels=["Normal", "Anomaly"],
        cbar=False,
    )
    plt.title("Confusion Matrix: LSTM-Autoencoder (Primary)", fontweight="bold")
    plt.xlabel("Predicted Label")
    plt.ylabel("Ground Truth Label")
    plt.savefig(os.path.join(fig_dir, "confusion_matrix.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate and Compare Anomaly Detection Models.")
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Path to config.yaml"
    )
    args = parser.parse_args()

    evaluate_all_models(config_path=args.config)
