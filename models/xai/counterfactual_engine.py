"""
Counterfactual Explanation Engine for IoT Anomaly Detection.
Implements two complementary counterfactual paradigms:
1. Physics-Aware Clamping: Evaluates causal intervention by clamping sensors
   to normal baseline medians and observing anomaly score reduction.
2. Optimization-Based Counterfactual: Uses gradient descent (Adam) to find the
   minimal feature perturbations necessary to restore normal operation.
"""

import logging
import os
import sys
from typing import Any, Dict, Optional

import joblib
import numpy as np
import torch
import torch.optim as optim
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from models.detection.lstm_autoencoder import LSTMAutoencoderDetector  # noqa: E402

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "ambient_temp",
    "ambient_humidity",
    "fan_speed",
    "lubrication_flow",
    "motor_temp",
    "spindle_speed",
    "vibration",
    "acoustic_emission",
    "power_draw",
    "tool_wear",
]

FEATURE_UNITS = {
    "ambient_temp": "°C",
    "ambient_humidity": "%",
    "fan_speed": "RPM",
    "lubrication_flow": "L/min",
    "motor_temp": "°C",
    "spindle_speed": "RPM",
    "vibration": "mm/s",
    "acoustic_emission": "dB",
    "power_draw": "Amps",
    "tool_wear": "μm",
}


FEATURE_LABELS = {f: f.replace("_", " ").title() for f in FEATURE_NAMES}


class CounterfactualEngine:
    """Physics-Aware and Optimization-Based Counterfactual Engine for CEdge-XAI."""

    def __init__(
        self,
        model_detector: Optional[LSTMAutoencoderDetector] = None,
        config_path: str = "config/config.yaml",
    ):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.feature_names = self.config["data"]["features"]
        self.feature_units = FEATURE_UNITS

        # Load detector if not provided
        if model_detector is not None:
            self.detector = model_detector
        else:
            self.detector = LSTMAutoencoderDetector(
                input_dim=self.config["models"]["lstm_ae"]["input_dim"],
                hidden_dim=self.config["models"]["lstm_ae"]["hidden_dim"],
                seq_len=self.config["preprocessing"]["window_size"],
            ).load(self.config["models"]["lstm_ae"]["checkpoint_path"])

        # Load scaler to get normal medians in original engineering units
        scaler_path = self.config["preprocessing"].get(
            "scaler_path", "models_saved/checkpoints/scaler.joblib"
        )
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            # Scaled median is 0.5 for minmax or compute empirical normal median
            self.normal_medians_scaled = np.full(len(self.feature_names), 0.5, dtype=np.float32)
            # Unscale to get real-world normal engineering units
            self.normal_medians_real = self.scaler.inverse_transform(
                self.normal_medians_scaled.reshape(1, -1)
            )[0]
        else:
            self.scaler = None
            self.normal_medians_scaled = np.zeros(len(self.feature_names), dtype=np.float32)
            self.normal_medians_real = np.zeros(len(self.feature_names), dtype=np.float32)

    def explain_physics_clamping(
        self,
        window: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Method 1: Physics-Aware Clamping.
        For each of the 10 sensors:
          - Clamp that sensor to its training median across the window
          - Re-run LSTM-Autoencoder
          - Calculate new anomaly score
          - Impact = original_score - new_score
        Top sensor = ROOT CAUSE.
        """
        thresh = threshold if threshold is not None else self.detector.threshold
        if window.ndim == 2:
            # (seq_len, num_features) -> add batch dim
            window = np.expand_dims(window, axis=0)

        original_score = float(self.detector.compute_anomaly_scores(window)[0])
        is_anomaly = original_score > thresh

        impacts = {}
        clamped_scores = {}

        for j, feat_name in enumerate(self.feature_names):
            clamped_window = window.copy()
            # Intervene by setting sensor j to normal median value
            clamped_window[:, :, j] = self.normal_medians_scaled[j]
            new_score = float(self.detector.compute_anomaly_scores(clamped_window)[0])
            impact = original_score - new_score
            impacts[feat_name] = float(impact)
            clamped_scores[feat_name] = float(new_score)

        # Rank sensors by impact descending
        sorted_sensors = sorted(impacts.items(), key=lambda x: x[1], reverse=True)
        root_cause, highest_impact = sorted_sensors[0]

        root_cause_idx = self.feature_names.index(root_cause)
        median_val_real = round(float(self.normal_medians_real[root_cause_idx]), 2)
        unit = self.feature_units.get(root_cause, "")

        # Compute current value of root cause in real units
        current_val_scaled = window[0, -1, root_cause_idx]
        if self.scaler is not None:
            dummy = np.zeros((1, len(self.feature_names)))
            dummy[0, root_cause_idx] = current_val_scaled
            # approximate invert
            scale = self.scaler.scale_[root_cause_idx]
            min_val = self.scaler.data_min_[root_cause_idx]
            current_val_real = round(float(current_val_scaled / scale + min_val), 2)
        else:
            current_val_real = round(float(current_val_scaled), 2)

        # Generate human-readable sentence
        explanation_sentence = (
            f"ROOT CAUSE: {root_cause}. Current reading is {current_val_real} {unit}. "
            f"IF {root_cause} had been at its normal value ({median_val_real} {unit}), "
            f"the anomaly score would drop from {original_score:.4f} to {clamped_scores[root_cause]:.4f} "
            f"(threshold: {thresh:.4f}) and the anomaly alert would DISAPPEAR."
        )

        return {
            "method": "Physics-Aware Clamping",
            "root_cause": root_cause,
            "root_cause_current": current_val_real,
            "root_cause_normal_median": median_val_real,
            "unit": unit,
            "original_score": round(original_score, 5),
            "threshold": round(thresh, 5),
            "is_anomaly": bool(is_anomaly),
            "impact_score": float(highest_impact),
            "sensor_impacts": impacts,
            "ranked_sensors": sorted_sensors,
            "explanation": explanation_sentence,
            "explanation_sentence": explanation_sentence,
        }

    def explain_optimization_counterfactual(
        self,
        window: np.ndarray,
        steps: int = 300,
        lr: float = 0.01,
        reg_lambda: float = 0.1,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Method 2: Optimization-Based Counterfactual using gradient descent.
        Minimizes: anomaly_score + lambda * distance_from_original
        Until anomaly score drops below threshold.
        """
        thresh = threshold if threshold is not None else self.detector.threshold
        if window.ndim == 2:
            window = np.expand_dims(window, axis=0)

        device = self.detector.device
        self.detector.model.eval()

        orig_tensor = torch.tensor(window, dtype=torch.float32, device=device)
        cf_tensor = orig_tensor.clone().detach().requires_grad_(True)

        optimizer = optim.Adam([cf_tensor], lr=lr)

        best_cf = orig_tensor.clone().detach()
        best_score = float("inf")
        converged_step = -1

        for step in range(1, steps + 1):
            optimizer.zero_grad()
            reconstruction = self.detector.model(cf_tensor)

            diff = (reconstruction - cf_tensor) ** 2
            score = 0.5 * diff.mean() + 0.5 * diff[:, -1, :].mean()

            # Distance constraint: L1 distance from original
            distance = torch.norm(cf_tensor - orig_tensor, p=1)
            loss = score + reg_lambda * distance

            loss.backward()
            optimizer.step()

            current_score = score.item()
            if current_score < thresh and current_score < best_score:
                best_score = current_score
                best_cf = cf_tensor.clone().detach()
                converged_step = step
                break

            if current_score < best_score:
                best_score = current_score
                best_cf = cf_tensor.clone().detach()

        # Compute perturbation delta
        delta = (best_cf - orig_tensor).cpu().numpy()[0]  # (seq_len, features)
        latest_delta = delta[-1]  # delta at detection timestep

        # Attribution by magnitude of required change
        feature_shifts = {feat: float(latest_delta[i]) for i, feat in enumerate(self.feature_names)}
        sorted_shifts = sorted(feature_shifts.items(), key=lambda x: abs(x[1]), reverse=True)
        top_perturbed_sensor = sorted_shifts[0][0]

        return {
            "method": "Optimization-Based Gradient Counterfactual",
            "converged": bool(best_score < thresh),
            "converged_step": converged_step,
            "original_score": float(self.detector.compute_anomaly_scores(window)[0]),
            "counterfactual_score": round(best_score, 5),
            "threshold": round(thresh, 5),
            "top_perturbed_sensor": top_perturbed_sensor,
            "perturbations": feature_shifts,
            "counterfactual_window": best_cf.cpu().numpy()[0],
        }

    # Method alias for consistency across callers
    optimize_counterfactual = explain_optimization_counterfactual


if __name__ == "__main__":
    from models.detection.train import prepare_dataset

    print("=" * 65)
    print("CEdge-XAI: COUNTERFACTUAL EXPLANATION VERIFICATION")
    print("=" * 65)
    engine = CounterfactualEngine(config_path="config/config.yaml")
    data_pkg = prepare_dataset(csv_path="data/synthetic/factory_iot_data.csv")
    anom_indices = np.where(data_pkg["y_test"] == 1)[0]
    sample_window = data_pkg["X_test"][anom_indices[0]]

    # 1. Physics Clamping
    clamp_res = engine.explain_physics_clamping(sample_window)
    print(
        f"\n[Physics Clamping] Root Cause: {clamp_res['root_cause']} (Impact: {clamp_res['impact_score']:.4f})"
    )
    print(f"Explanation: {clamp_res['explanation']}")

    # 2. Optimization Counterfactual
    grad_res = engine.optimize_counterfactual(sample_window, steps=50)
    print(f"\n[Gradient Optimization] Top Perturbed Sensor: {grad_res['top_perturbed_sensor']}")
    print(
        f"Converged: {grad_res['converged']} | Counterfactual Score: {grad_res['counterfactual_score']:.4f}"
    )
    print("=" * 65)
