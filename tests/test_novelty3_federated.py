"""
Unit and Integration Tests for Novelty 3: Privacy-Preserving Federated Training.
"""

import os
import sys
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.detection.lstm_autoencoder import LSTMAutoencoder
from models.federated.federated_experiment import (
    evaluate_model_on_data,
    partition_non_iid_factories,
    verify_federated_privacy,
)
from models.federated.fl_utils import get_model_parameters, set_model_parameters


def test_non_iid_partitioning():
    partitions = partition_non_iid_factories(
        csv_path="data/synthetic/factory_iot_data.csv",
        window_size=60,
    )
    assert len(partitions) == 3
    assert "Factory_A" in partitions
    assert "Factory_B" in partitions
    assert "Factory_C" in partitions

    for name, p in partitions.items():
        assert p["X_train_normal"].ndim == 3
        assert p["X_train_normal"].shape[1] == 60
        assert p["X_train_normal"].shape[2] == 10
        assert len(p["scenarios"]) >= 1


def test_programmatic_privacy_verification():
    # Valid payload containing only model weights
    model = LSTMAutoencoder(input_dim=10, hidden_dim=32, seq_len=60, num_layers=2)
    valid_params = get_model_parameters(model)

    clean_payloads = [
        {"client_name": "Factory_A", "parameters": valid_params, "num_samples": 100},
        {"client_name": "Factory_B", "parameters": valid_params, "num_samples": 100},
    ]

    df_audit = verify_federated_privacy(clean_payloads, log_csv_path=None)
    assert df_audit["Privacy_Audit_Passed"].all()

    # Leaked payload with raw sensor data
    dirty_payloads = [
        {
            "client_name": "Factory_A",
            "parameters": valid_params,
            "num_samples": 100,
            "raw_data": np.random.randn(10, 10),  # FORBIDDEN LEAK
        }
    ]

    with pytest.raises(AssertionError, match="PRIVACY LEAK DETECTED"):
        verify_federated_privacy(dirty_payloads, log_csv_path=None)


def test_model_evaluation_metrics():
    model = LSTMAutoencoder(input_dim=10, hidden_dim=32, seq_len=60, num_layers=2)
    dummy_x = np.random.randn(20, 60, 10).astype(np.float32)
    dummy_y = np.zeros(20, dtype=int)
    dummy_y[:5] = 1

    metrics = evaluate_model_on_data(model, dummy_x, dummy_y)
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert 0.0 <= metrics["f1"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
