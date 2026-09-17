"""
Training pipeline for all 4 anomaly detection models on IoT time-series data.
Handles data loading, train/val/test split, windowing, threshold calibration,
and checkpoint saving.
"""

import argparse
import logging
import os
import sys
from typing import Any, Dict, Tuple

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.preprocessing import MinMaxScaler  # noqa: E402
import yaml  # noqa: E402

from models.detection.anomaly_transformer_model import AnomalyTransformerDetector  # noqa: E402
from models.detection.isolation_forest_model import IsolationForestDetector  # noqa: E402
from models.detection.lstm_autoencoder import LSTMAutoencoderDetector  # noqa: E402
from models.detection.one_class_svm_model import OneClassSVMDetector  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
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
DEFAULT_FEATURES = FEATURE_COLUMNS


def create_sliding_windows(
    data: np.ndarray,
    labels: np.ndarray,
    window_size: int = 60,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate sliding windows for sequence models.
    Each window covers [i : i + window_size].
    Label corresponds to whether the final timestep in the window is anomalous.
    """
    num_windows = (len(data) - window_size) // stride + 1
    X = []
    y = []
    for i in range(0, num_windows * stride, stride):
        window_data = data[i : i + window_size]
        window_label = labels[i + window_size - 1]  # Online detection target at current step
        X.append(window_data)
        y.append(window_label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=int)


def prepare_dataset(
    csv_path: str = "data/synthetic/factory_iot_data.csv",
    window_size: int = 60,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    scaler_save_path: str = "models_saved/checkpoints/scaler.joblib",
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Split time series, fit MinMaxScaler on normal training data, and generate sequences.
    """
    logger.info(f"Loading IoT data from {csv_path}...")
    df = pd.read_csv(csv_path)

    feat_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    X_raw = df[feat_cols].values
    y_raw = df["anomaly"].values if "anomaly" in df.columns else np.zeros(len(df), dtype=int)

    # Fit scaler strictly on the first 70% normal readings
    n_total = len(df)
    n_train_raw = int(n_total * train_ratio)
    scaler_train_mask = y_raw[:n_train_raw] == 0
    scaler = MinMaxScaler()
    scaler.fit(X_raw[:n_train_raw][scaler_train_mask])

    os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
    joblib.dump(scaler, scaler_save_path)
    logger.info(f"Fitted and saved MinMaxScaler to {scaler_save_path}")

    X_scaled = scaler.transform(X_raw)

    # Create continuous temporal sliding windows across the full dataset
    X_all_seq, y_all_seq = create_sliding_windows(X_scaled, y_raw, window_size=window_size)

    # Pure normal mask: entire window contains zero anomaly timesteps
    pure_norm_mask = np.array(
        [y_raw[i : i + window_size].sum() == 0 for i in range(len(X_all_seq))]
    )
    norm_indices = np.where(pure_norm_mask)[0]
    anom_indices = np.where(y_all_seq == 1)[0]

    rng = np.random.RandomState(random_state)
    rng.shuffle(norm_indices)
    rng.shuffle(anom_indices)

    n_norm_train = int(len(norm_indices) * train_ratio)
    n_norm_val = int(len(norm_indices) * val_ratio)
    train_norm_idx = norm_indices[:n_norm_train]
    val_norm_idx = norm_indices[n_norm_train : n_norm_train + n_norm_val]
    test_norm_idx = norm_indices[n_norm_train + n_norm_val :]

    n_anom_val = len(anom_indices) // 2
    val_idx = np.concatenate([val_norm_idx, anom_indices[:n_anom_val]])
    test_idx = np.concatenate([test_norm_idx, anom_indices[n_anom_val:]])
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    X_train_normal, X_val, y_val = X_all_seq[train_norm_idx], X_all_seq[val_idx], y_all_seq[val_idx]
    X_test, y_test = X_all_seq[test_idx], y_all_seq[test_idx]

    logger.info(
        f"Data prepared: Train={X_train_normal.shape}, Val={X_val.shape} (Anom={y_val.sum()}), Test={X_test.shape} (Anom={y_test.sum()})"
    )
    return {
        "X_train_normal": X_train_normal,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "scaler": scaler,
        "feature_names": feat_cols,
    }


def optimize_threshold(
    scores: np.ndarray, y_true: np.ndarray, n_thresholds: int = 100
) -> Tuple[float, float]:
    """Find threshold that maximizes F1 score on validation set."""
    if y_true.sum() == 0:
        return float(np.percentile(scores, 95)), 0.0

    percentiles = np.linspace(1, 99.9, n_thresholds)
    threshold_candidates = np.percentile(scores, percentiles)

    best_thresh = float(threshold_candidates[0])
    best_f1 = -1.0

    for th in threshold_candidates:
        preds = (scores > th).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(th)

    return best_thresh, best_f1


def train_all_models(
    config_path: str = "configs/config.yaml",
    force_retrain: bool = False,
) -> Dict[str, Any]:
    """Train all 4 models and save checkpoints."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_pkg = prepare_dataset(
        csv_path=config["data"]["synthetic_csv"],
        window_size=config["preprocessing"]["window_size"],
        train_ratio=config["preprocessing"]["train_ratio"],
        val_ratio=config["preprocessing"]["val_ratio"],
        test_ratio=config["preprocessing"]["test_ratio"],
    )

    X_train_normal = data_pkg["X_train_normal"]
    X_val = data_pkg["X_val"]
    y_val = data_pkg["y_val"]

    models_dict = {}

    # 1. Isolation Forest
    logger.info("=== 1/4 Training Isolation Forest ===")
    cfg_if = config["models"]["isolation_forest"]
    if_det = IsolationForestDetector(
        n_estimators=cfg_if["n_estimators"],
        contamination=cfg_if["contamination"],
        checkpoint_path=cfg_if["checkpoint_path"],
    )
    if_det.fit(X_train_normal)
    opt_th_if, val_f1_if = optimize_threshold(if_det.compute_anomaly_scores(X_val), y_val)
    if_det.threshold = opt_th_if
    if_det.save()
    models_dict["Isolation Forest"] = if_det
    logger.info(f"Isolation Forest Optimal Threshold={opt_th_if:.5f}, Val F1={val_f1_if:.4f}")

    # 2. One-Class SVM
    logger.info("=== 2/4 Training One-Class SVM ===")
    cfg_svm = config["models"]["one_class_svm"]
    svm_det = OneClassSVMDetector(
        kernel=cfg_svm["kernel"],
        gamma=cfg_svm["gamma"],
        nu=cfg_svm["nu"],
        checkpoint_path=cfg_svm["checkpoint_path"],
    )
    svm_det.fit(X_train_normal)
    opt_th_svm, val_f1_svm = optimize_threshold(svm_det.compute_anomaly_scores(X_val), y_val)
    svm_det.threshold = opt_th_svm
    svm_det.save()
    models_dict["One-Class SVM"] = svm_det
    logger.info(f"One-Class SVM Optimal Threshold={opt_th_svm:.5f}, Val F1={val_f1_svm:.4f}")

    # 3. LSTM-Autoencoder (Primary Model)
    logger.info("=== 3/4 Training LSTM-Autoencoder (Primary) ===")
    cfg_lstm = config["models"]["lstm_ae"]
    lstm_det = LSTMAutoencoderDetector(
        input_dim=cfg_lstm["input_dim"],
        hidden_dim=cfg_lstm["hidden_dim"],
        seq_len=config["preprocessing"]["window_size"],
        num_layers=cfg_lstm["num_layers"],
        dropout=cfg_lstm["dropout"],
        lr=cfg_lstm["learning_rate"],
        checkpoint_path=cfg_lstm["checkpoint_path"],
    )
    lstm_det.fit(
        X_train=X_train_normal,
        X_val=X_val[y_val == 0],
        epochs=cfg_lstm["epochs"],
        batch_size=cfg_lstm["batch_size"],
        patience=cfg_lstm["patience"],
    )
    opt_th_lstm, val_f1_lstm = optimize_threshold(lstm_det.compute_anomaly_scores(X_val), y_val)
    lstm_det.threshold = opt_th_lstm
    lstm_det.save()
    meta = {
        "threshold": float(opt_th_lstm),
        "val_f1": float(val_f1_lstm),
        "input_dim": cfg_lstm["input_dim"],
        "hidden_dim": cfg_lstm["hidden_dim"],
        "window_size": config["preprocessing"]["window_size"],
        "features": FEATURE_COLUMNS,
    }
    with open("models_saved/checkpoints/lstm_ae_meta.yaml", "w") as f:
        yaml.dump(meta, f)
    models_dict["LSTM-Autoencoder"] = lstm_det
    logger.info(f"LSTM-Autoencoder Optimal Threshold={opt_th_lstm:.6f}, Val F1={val_f1_lstm:.4f}")

    # 4. Anomaly Transformer
    logger.info("=== 4/4 Training Anomaly Transformer ===")
    cfg_at = config["models"]["anomaly_transformer"]
    at_det = AnomalyTransformerDetector(
        input_dim=cfg_at["input_dim"],
        d_model=cfg_at["d_model"],
        n_heads=cfg_at["n_heads"],
        seq_len=config["preprocessing"]["window_size"],
        num_layers=cfg_at["num_layers"],
        lr=cfg_at["learning_rate"],
        checkpoint_path=cfg_at["checkpoint_path"],
    )
    at_det.fit(X_train=X_train_normal, epochs=cfg_at["epochs"], batch_size=cfg_at["batch_size"])
    opt_th_at, val_f1_at = optimize_threshold(at_det.compute_anomaly_scores(X_val), y_val)
    at_det.threshold = opt_th_at
    at_det.save()
    models_dict["Anomaly Transformer"] = at_det
    logger.info(f"Anomaly Transformer Optimal Threshold={opt_th_at:.6f}, Val F1={val_f1_at:.4f}")

    logger.info("All 4 models successfully trained, calibrated, and saved.")
    return models_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 4 Anomaly Detection Models.")
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Path to config.yaml"
    )
    args = parser.parse_args()

    train_all_models(config_path=args.config)
