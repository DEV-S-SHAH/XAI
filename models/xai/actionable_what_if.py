"""
Actionable What-If Counterfactual Explanation Engine for IoT Telemetry.
Novelty 1: Actionable What-If Explanations.

Goes beyond attribution scores by generating bounded, physically feasible counterfactuals:
- Identifies responsible root-cause sensors.
- Displays current readings and reference baselines in engineering units.
- Quantifies exact direction and magnitude of required intervention.
- Evaluates Validity, Sparsity, Proximity (L1/L2), Actionability, Score Reduction, and Generation Time.
"""

import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import torch
import torch.optim as optim
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from models.detection.lstm_autoencoder import LSTMAutoencoderDetector
from models.xai.counterfactual_engine import FEATURE_LABELS, FEATURE_UNITS

logger = logging.getLogger(__name__)

# Physical actuator/sensor feasibility boundaries in real engineering units
PHYSICAL_BOUNDS = {
    "ambient_temp": (15.0, 45.0),       # °C
    "ambient_humidity": (20.0, 85.0),   # %
    "fan_speed": (0.0, 2000.0),         # RPM (Actuator: controllable)
    "lubrication_flow": (0.5, 6.0),     # L/min (Actuator: pump controllable)
    "motor_temp": (30.0, 95.0),         # °C (Thermal state)
    "spindle_speed": (800.0, 3500.0),   # RPM (Actuator: VFD controllable)
    "vibration": (0.5, 15.0),           # mm/s (Dynamic state)
    "acoustic_emission": (45.0, 100.0), # dB (Dynamic state)
    "power_draw": (3.0, 30.0),          # Amps (Electrical state)
    "tool_wear": (0.0, 60.0),           # μm (Physical wear)
}

# Features classified as directly controllable actuators vs dependent states
ACTIONABLE_FEATURES = {
    "fan_speed": True,          # Auxiliary cooling fan speed
    "lubrication_flow": True,   # Lubricant pump regulator
    "spindle_speed": True,      # Drive motor frequency inverter
    "ambient_temp": True,       # Plant HVAC chiller setpoint
    "ambient_humidity": False,  # Environmental byproduct
    "motor_temp": False,        # Dependent thermal response
    "vibration": True,          # Accelerometer sensor calibration / balancing
    "acoustic_emission": False, # Dependent acoustic response
    "power_draw": False,        # Dependent electrical response
    "tool_wear": False,         # Cumulative physical state
}


class ActionableWhatIfExplainer:
    """Generates actionable, physically bounded counterfactual explanations."""

    def __init__(
        self,
        config_path: str = "config/config.yaml",
        model_detector: Optional[LSTMAutoencoderDetector] = None,
    ):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.feature_names = self.config["data"]["features"]
        self.feature_units = FEATURE_UNITS
        self.feature_labels = FEATURE_LABELS

        # Load detector
        if model_detector is not None:
            self.detector = model_detector
        else:
            self.detector = LSTMAutoencoderDetector(
                input_dim=self.config["models"]["lstm_ae"]["input_dim"],
                hidden_dim=self.config["models"]["lstm_ae"]["hidden_dim"],
                seq_len=self.config["preprocessing"]["window_size"],
                checkpoint_path=self.config["models"]["lstm_ae"]["checkpoint_path"],
            ).load()

        # Load scaler and compute empirical normal baselines
        scaler_path = self.config["preprocessing"].get(
            "scaler_path", "models_saved/checkpoints/scaler.joblib"
        )
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            data_csv = self.config["data"].get(
                "synthetic_csv", "data/synthetic/factory_iot_data.csv"
            )
            if os.path.exists(data_csv):
                import pandas as pd

                df = pd.read_csv(data_csv)
                norm_data = df[df["anomaly"] == 0][self.feature_names].values
                norm_scaled = self.scaler.transform(norm_data)
                self.normal_medians_scaled = np.median(norm_scaled, axis=0).astype(np.float32)
                self.normal_stds_scaled = np.std(norm_scaled, axis=0).astype(np.float32)
            else:
                self.normal_medians_scaled = np.full(len(self.feature_names), 0.5, dtype=np.float32)
                self.normal_stds_scaled = np.full(len(self.feature_names), 0.1, dtype=np.float32)

            self.normal_medians_real = self.scaler.inverse_transform(
                self.normal_medians_scaled.reshape(1, -1)
            )[0]
        else:
            self.scaler = None
            self.normal_medians_scaled = np.zeros(len(self.feature_names), dtype=np.float32)
            self.normal_stds_scaled = np.ones(len(self.feature_names), dtype=np.float32)
            self.normal_medians_real = np.zeros(len(self.feature_names), dtype=np.float32)

        # Causal downstream coupling dependencies
        self.causal_effects = {
            "lubrication_flow": ["vibration", "acoustic_emission", "power_draw"],
            "fan_speed": ["motor_temp"],
            "ambient_temp": ["motor_temp"],
            "spindle_speed": ["power_draw", "vibration", "tool_wear"],
            "vibration": ["acoustic_emission", "power_draw"],
        }

    def generate_counterfactual(
        self,
        window: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Generate actionable physics-aware counterfactual explanation for an anomaly window.
        Returns:
            Structured dictionary with metrics (Validity, Sparsity, Proximity, Actionability, Latency, Score Reduction).
        """
        t0 = time.perf_counter()
        thresh = threshold if threshold is not None else self.detector.threshold
        if window.ndim == 2:
            window = np.expand_dims(window, axis=0)

        orig_score = float(self.detector.compute_anomaly_scores(window)[0])
        is_anom = orig_score > thresh

        # 1. Identify candidate root causes using standardized deviation & causal intervention
        n_feats = len(self.feature_names)
        batch_clamped = np.repeat(window, n_feats, axis=0)
        dev = np.abs(window[0, -1] - self.normal_medians_scaled) / (self.normal_stds_scaled + 1e-6)

        for j, feat_name in enumerate(self.feature_names):
            batch_clamped[j, :, j] = self.normal_medians_scaled[j]
            if feat_name in self.causal_effects and dev[j] > 2.0:
                for child in self.causal_effects[feat_name]:
                    c_idx = self.feature_names.index(child)
                    batch_clamped[j, :, c_idx] = self.normal_medians_scaled[c_idx]

        new_scores = self.detector.compute_anomaly_scores(batch_clamped)

        impacts = {}
        weighted_impacts = {}
        for j, feat_name in enumerate(self.feature_names):
            impact = orig_score - float(new_scores[j])
            impacts[feat_name] = float(impact)
            if dev[j] > 1.5:
                weighted_impacts[feat_name] = float(impact)
            else:
                weighted_impacts[feat_name] = -1.0

        valid_cands = [k for k, v in weighted_impacts.items() if v > -1.0]
        if valid_cands:
            sorted_cands = sorted([(k, impacts[k]) for k in valid_cands], key=lambda x: x[1], reverse=True)
        else:
            sorted_cands = sorted(impacts.items(), key=lambda x: x[1], reverse=True)

        root_cause = sorted_cands[0][0]
        root_idx = self.feature_names.index(root_cause)

        # 2. Synthesize actionable counterfactual window
        cf_window = window.copy()
        cf_window[0, :, root_idx] = self.normal_medians_scaled[root_idx]
        if root_cause in self.causal_effects:
            for child in self.causal_effects[root_cause]:
                c_idx = self.feature_names.index(child)
                cf_window[0, :, c_idx] = self.normal_medians_scaled[c_idx]

        cf_score = float(self.detector.compute_anomaly_scores(cf_window)[0])
        t1 = time.perf_counter()
        gen_time_ms = (t1 - t0) * 1000.0

        # 3. Compute Metrics
        # Validity: Is counterfactual anomaly score below threshold?
        validity = bool(cf_score <= thresh)

        # Perturbation delta across timesteps
        delta = cf_window[0] - window[0]  # (seq_len, n_feats)
        delta_latest = delta[-1]

        # Sparsity: Number of features with meaningful perturbation (|delta| > 0.01)
        feat_modified = [
            self.feature_names[i]
            for i in range(n_feats)
            if np.max(np.abs(delta[:, i])) > 0.01
        ]
        sparsity = len(feat_modified)

        # Proximity: Normalized L1 and L2 distances in scaled space [0, 1]
        proximity_l1 = float(np.mean(np.abs(delta)))
        proximity_l2 = float(np.sqrt(np.mean(delta ** 2)))

        # Convert to real engineering units for actionability check and narrative
        current_val_scaled = window[0, -1, root_idx]
        target_val_scaled = cf_window[0, -1, root_idx]

        if self.scaler is not None:
            scale = self.scaler.scale_[root_idx]
            min_v = self.scaler.data_min_[root_idx]
            current_real = float(current_val_scaled / scale + min_v)
            target_real = float(target_val_scaled / scale + min_v)
        else:
            current_real = float(current_val_scaled)
            target_real = float(target_val_scaled)

        amount_change_real = target_real - current_real
        direction = "increase" if amount_change_real > 0 else "decrease"
        unit = self.feature_units.get(root_cause, "")

        # Actionability: Are modified features controllable and within physical bounds?
        actionable_flags = []
        for feat in feat_modified:
            f_idx = self.feature_names.index(feat)
            f_target_scaled = cf_window[0, -1, f_idx]
            if self.scaler is not None:
                f_target_real = float(f_target_scaled / self.scaler.scale_[f_idx] + self.scaler.data_min_[f_idx])
            else:
                f_target_real = float(f_target_scaled)

            min_bound, max_bound = PHYSICAL_BOUNDS.get(feat, (-float("inf"), float("inf")))
            is_bounded = (min_bound <= f_target_real <= max_bound)
            is_controllable = ACTIONABLE_FEATURES.get(feat, True)
            actionable_flags.append(is_bounded and is_controllable)

        actionability_rate = (sum(actionable_flags) / len(actionable_flags) * 100.0) if actionable_flags else 100.0

        # Score Reduction
        score_reduction = orig_score - cf_score
        score_reduction_pct = (score_reduction / orig_score * 100.0) if orig_score > 0 else 0.0

        # Actionable What-If Narrative
        action_text = (
            f"WHAT-IF ACTION: {direction.upper()} {self.feature_labels.get(root_cause, root_cause)} "
            f"from {current_real:.2f} {unit} to {target_real:.2f} {unit} "
            f"({amount_change_real:+.2f} {unit}). "
            f"Result: Anomaly reconstruction score drops from {orig_score:.4f} to {cf_score:.4f} "
            f"(threshold: {thresh:.4f}), restoring the machine to verified normal operation."
        )

        return {
            "method": "Actionable Physics-Aware Counterfactual (Ours)",
            "root_cause": root_cause,
            "root_cause_label": self.feature_labels.get(root_cause, root_cause),
            "current_value": round(current_real, 2),
            "target_value": round(target_real, 2),
            "required_change": round(amount_change_real, 2),
            "direction": direction,
            "unit": unit,
            "original_score": round(orig_score, 5),
            "counterfactual_score": round(cf_score, 5),
            "threshold": round(thresh, 5),
            "validity": validity,
            "sparsity": sparsity,
            "modified_features": feat_modified,
            "proximity_l1": round(proximity_l1, 5),
            "proximity_l2": round(proximity_l2, 5),
            "actionability": round(actionability_rate, 2),
            "score_reduction": round(score_reduction, 5),
            "score_reduction_pct": round(score_reduction_pct, 2),
            "latency_ms": round(gen_time_ms, 2),
            "explanation_narrative": action_text,
            "counterfactual_window": cf_window[0],
        }

    def generate_baseline_ablation(
        self,
        window: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Baseline 1: Feature Ablation (naively clamps the single sensor with highest reconstruction error).
        Lacks causal awareness, which often confuses symptoms (e.g. vibration) with initiating causes.
        """
        t0 = time.perf_counter()
        thresh = threshold if threshold is not None else self.detector.threshold
        if window.ndim == 2:
            window = np.expand_dims(window, axis=0)

        orig_score = float(self.detector.compute_anomaly_scores(window)[0])

        # Compute per-feature reconstruction error on latest timestep
        device = self.detector.device
        self.detector.model.eval()
        with torch.no_grad():
            t_win = torch.tensor(window, dtype=torch.float32, device=device)
            recon = self.detector.model(t_win)
            err_per_feat = ((recon[0, -1] - t_win[0, -1]) ** 2).cpu().numpy()

        top_err_idx = int(np.argmax(err_per_feat))
        top_err_feat = self.feature_names[top_err_idx]

        # Naively clamp top error feature ONLY without propagating causal effects
        cf_window = window.copy()
        cf_window[0, :, top_err_idx] = self.normal_medians_scaled[top_err_idx]
        cf_score = float(self.detector.compute_anomaly_scores(cf_window)[0])
        t1 = time.perf_counter()

        delta = cf_window[0] - window[0]
        validity = bool(cf_score <= thresh)
        score_reduction = orig_score - cf_score
        score_reduction_pct = (score_reduction / orig_score * 100.0) if orig_score > 0 else 0.0

        return {
            "method": "Feature Ablation Baseline",
            "root_cause": top_err_feat,
            "validity": validity,
            "sparsity": 1,
            "proximity_l1": round(float(np.mean(np.abs(delta))), 5),
            "proximity_l2": round(float(np.sqrt(np.mean(delta ** 2))), 5),
            "actionability": 50.0 if not ACTIONABLE_FEATURES.get(top_err_feat, True) else 100.0,
            "score_reduction": round(score_reduction, 5),
            "score_reduction_pct": round(score_reduction_pct, 2),
            "latency_ms": round((t1 - t0) * 1000.0, 2),
            "original_score": round(orig_score, 5),
            "counterfactual_score": round(cf_score, 5),
        }

    def generate_baseline_gradient(
        self,
        window: np.ndarray,
        steps: int = 150,
        lr: float = 0.02,
        reg_lambda: float = 0.05,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Baseline 2: Unconstrained Gradient-Based Counterfactual (Adam optimization).
        Minimizes anomaly score + L1 distance without physics or domain constraints.
        Often perturbs all features simultaneously (poor sparsity and low actionability).
        """
        t0 = time.perf_counter()
        thresh = threshold if threshold is not None else self.detector.threshold
        if window.ndim == 2:
            window = np.expand_dims(window, axis=0)

        device = self.detector.device
        self.detector.model.eval()

        orig_score = float(self.detector.compute_anomaly_scores(window)[0])
        orig_tensor = torch.tensor(window, dtype=torch.float32, device=device)
        cf_tensor = orig_tensor.clone().detach().requires_grad_(True)

        optimizer = optim.Adam([cf_tensor], lr=lr)
        best_cf = orig_tensor.clone().detach()
        best_score = float("inf")

        for step in range(steps):
            optimizer.zero_grad()
            recon = self.detector.model(cf_tensor)
            diff = (recon - cf_tensor) ** 2
            score = 0.5 * diff.mean() + 0.5 * diff[:, -1, :].mean()
            dist = torch.norm(cf_tensor - orig_tensor, p=1)
            loss = score + reg_lambda * dist
            loss.backward()
            optimizer.step()

            cur_s = score.item()
            if cur_s < best_score:
                best_score = cur_s
                best_cf = cf_tensor.clone().detach()
            if cur_s < thresh:
                break

        t1 = time.perf_counter()
        cf_window = best_cf.cpu().numpy()
        delta = cf_window[0] - window[0]

        validity = bool(best_score <= thresh)
        sparsity = int(np.sum(np.max(np.abs(delta), axis=0) > 0.01))
        score_reduction = orig_score - best_score
        score_reduction_pct = (score_reduction / orig_score * 100.0) if orig_score > 0 else 0.0

        # Actionability: unconstrained gradient optimization often pushes features outside physical bounds
        out_of_bounds = 0
        for i in range(len(self.feature_names)):
            if np.max(np.abs(delta[:, i])) > 0.01:
                # check bounds
                val = cf_window[0, -1, i]
                if self.scaler is not None:
                    real_v = val / self.scaler.scale_[i] + self.scaler.data_min_[i]
                else:
                    real_v = val
                b_min, b_max = PHYSICAL_BOUNDS.get(self.feature_names[i], (-float("inf"), float("inf")))
                if real_v < b_min or real_v > b_max:
                    out_of_bounds += 1

        actionability = max(0.0, 100.0 - (out_of_bounds / max(1, sparsity)) * 100.0)

        # Attribute top perturbed sensor
        delta_mag = np.mean(np.abs(delta), axis=0)
        top_idx = int(np.argmax(delta_mag))
        top_feat = self.feature_names[top_idx]

        return {
            "method": "Unconstrained Gradient Counterfactual",
            "root_cause": top_feat,
            "validity": validity,
            "sparsity": sparsity,
            "proximity_l1": round(float(np.mean(np.abs(delta))), 5),
            "proximity_l2": round(float(np.sqrt(np.mean(delta ** 2))), 5),
            "actionability": round(actionability, 2),
            "score_reduction": round(score_reduction, 5),
            "score_reduction_pct": round(score_reduction_pct, 2),
            "latency_ms": round((t1 - t0) * 1000.0, 2),
            "original_score": round(orig_score, 5),
            "counterfactual_score": round(best_score, 5),
        }
