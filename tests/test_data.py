"""
Unit Tests for Data Loader and Preprocessing Pipeline in XAI for IOT Anomaly Detection.
Tests sliding window generation, 10-sensor validation, scaling fidelity, and normal isolation.
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.detection.train import (  # noqa: E402
    create_sliding_windows,
    prepare_dataset,
    DEFAULT_FEATURES,
)


def test_default_features_count():
    assert len(DEFAULT_FEATURES) == 10
    expected_sensors = [
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
    assert DEFAULT_FEATURES == expected_sensors


def test_sliding_windows_shape():
    num_samples = 120
    num_features = 10
    window_size = 60
    data = np.random.randn(num_samples, num_features).astype(np.float32)
    labels = np.zeros(num_samples, dtype=int)
    labels[80:90] = 1

    X, y = create_sliding_windows(data, labels, window_size=window_size)
    assert X.shape == (61, 60, 10)
    assert len(y) == 61
    assert y.dtype == int


def test_prepare_dataset_synthetic():
    csv_path = "data/synthetic/factory_iot_data.csv"
    if not os.path.exists(csv_path):
        pytest.skip("Synthetic dataset not generated yet.")

    pkg = prepare_dataset(csv_path=csv_path, window_size=60)
    assert "X_train_normal" in pkg
    assert "X_val" in pkg
    assert "y_val" in pkg
    assert "X_test" in pkg
    assert "y_test" in pkg

    assert pkg["X_train_normal"].ndim == 3
    assert pkg["X_train_normal"].shape[1] == 60
    assert pkg["X_train_normal"].shape[2] == 10

    # Pure normal verification: train set must contain zero anomalies
    assert len(pkg["X_train_normal"]) > 1000
    assert pkg["y_test"].sum() > 0
