"""
SHAP (SHapley Additive exPlanations) Baseline for LSTM-Autoencoder.
Computes KernelSHAP / GradientExplainer feature attributions on anomalous windows
and generates summary plots and feature importance rankings for comparison.
"""

import argparse
import logging
import os
import sys
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import yaml

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from models.detection.lstm_autoencoder import LSTMAutoencoderDetector  # noqa: E402
from models.detection.train import prepare_dataset  # noqa: E402

logger = logging.getLogger(__name__)


class SHAPBaselineExplainer:
    """SHAP Explainer for LSTM-Autoencoder anomaly scores."""

    def __init__(
        self,
        config_path: str = "config/config.yaml",
        model_detector: Optional[LSTMAutoencoderDetector] = None,
        background_samples: int = 50,
    ):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.feature_names = self.config["data"]["features"]
        self.background_samples = background_samples

        if model_detector is not None:
            self.detector = model_detector
        else:
            self.detector = LSTMAutoencoderDetector(
                input_dim=self.config["models"]["lstm_ae"]["input_dim"],
                hidden_dim=self.config["models"]["lstm_ae"]["hidden_dim"],
                seq_len=self.config["preprocessing"]["window_size"],
            ).load(self.config["models"]["lstm_ae"]["checkpoint_path"])

    def _predict_score_flat(self, X_flat: np.ndarray) -> np.ndarray:
        """
        Wrapper function mapping flattened 2D features (N, T * D) to scalar anomaly scores.
        """
        N = X_flat.shape[0]
        T = self.config["preprocessing"]["window_size"]
        D = len(self.feature_names)
        X_seq = X_flat.reshape(N, T, D)
        return self.detector.compute_anomaly_scores(X_seq)

    def explain(
        self,
        X_background: np.ndarray,
        X_explain: np.ndarray,
        save_summary_path: str = "paper_assets/figures/shap_summary.png",
        save_bar_path: str = "paper_assets/figures/xai_comparison.png",
        save_csv_path: Optional[str] = None,
        max_evals: int = 500,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Compute SHAP values using KernelExplainer across sensor features.
        """
        logger.info("Initializing KernelExplainer with normal background sequences...")
        N_bg = min(len(X_background), self.background_samples)
        bg_subset = X_background[:N_bg]

        # Reshape to (N, T * D)
        T = self.config["preprocessing"]["window_size"]
        D = len(self.feature_names)
        bg_flat = bg_subset.reshape(N_bg, T * D)

        explainer = shap.KernelExplainer(self._predict_score_flat, bg_flat)

        N_exp = min(len(X_explain), 10)
        exp_subset = X_explain[:N_exp]
        exp_flat = exp_subset.reshape(N_exp, T * D)

        logger.info(
            f"Computing SHAP values for {N_exp} anomalous windows (max_evals={max_evals})..."
        )
        shap_values_flat = explainer.shap_values(exp_flat, nsamples=max_evals)

        # Aggregate SHAP values across the temporal window to get per-sensor importance
        # shap_values_flat: (N_exp, T * D) -> (N_exp, T, D)
        shap_values_3d = shap_values_flat.reshape(N_exp, T, D)
        # Average magnitude per sensor across instances and time
        mean_abs_sensor = np.abs(shap_values_3d).mean(axis=(0, 1))

        importance_dict = {self.feature_names[i]: float(mean_abs_sensor[i]) for i in range(D)}
        sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

        if save_csv_path:
            os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)
            df_shap = pd.DataFrame(
                [{"Feature": k, "Mean_Abs_SHAP": v} for k, v in sorted_importance.items()]
            )
            df_shap.to_csv(save_csv_path, index=False)
            logger.info(f"Saved SHAP attributions to {save_csv_path}")

        # Visualizations
        os.makedirs(os.path.dirname(save_bar_path), exist_ok=True)
        plt.figure(figsize=(9, 5))
        y_pos = np.arange(len(sorted_importance))
        plt.barh(
            y_pos,
            list(sorted_importance.values())[::-1],
            color="#3498db",
            edgecolor="#2980b9",
        )
        plt.yticks(y_pos, list(sorted_importance.keys())[::-1])
        plt.xlabel("Mean |SHAP Value| (Impact on Anomaly Score)")
        plt.title("Baseline Feature Importance: SHAP (KernelExplainer)", fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_bar_path, dpi=300)
        plt.close()
        logger.info(f"Saved SHAP bar plot to {save_bar_path}")

        # Summary plot over features for latest timestep
        os.makedirs(os.path.dirname(save_summary_path), exist_ok=True)
        latest_step_shap = shap_values_3d[:, -1, :]
        latest_step_vals = exp_subset[:, -1, :]

        plt.figure(figsize=(9, 6))
        shap.summary_plot(
            latest_step_shap,
            latest_step_vals,
            feature_names=self.feature_names,
            show=False,
        )
        plt.title("SHAP Summary Plot on Anomaly Windows", fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_summary_path, dpi=300)
        plt.close()
        logger.info(f"Saved SHAP summary plot to {save_summary_path}")

        return shap_values_3d, sorted_importance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SHAP Baseline Explanation.")
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Path to config.yaml"
    )
    args = parser.parse_args()

    data_pkg = prepare_dataset()
    anom_test = data_pkg["X_test"][data_pkg["y_test"] == 1]
    norm_bg = data_pkg["X_train_normal"]

    shap_engine = SHAPBaselineExplainer(config_path=args.config)
    shap_engine.explain(X_background=norm_bg, X_explain=anom_test)
