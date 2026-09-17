"""
Unit Tests for Detection Models in CEdge-XAI.
Tests LSTM-AE, Isolation Forest, One-Class SVM, Anomaly Transformer, and ONNX Runtime.
"""

import os
import sys
import numpy as np
import onnxruntime as ort
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.detection.anomaly_transformer_model import AnomalyTransformer  # noqa: E402
from models.detection.isolation_forest_model import IsolationForestDetector  # noqa: E402
from models.detection.lstm_autoencoder import LSTMAutoencoder, LSTMAutoencoderDetector  # noqa: E402
from models.detection.one_class_svm_model import OneClassSVMDetector  # noqa: E402


def test_lstm_autoencoder_forward_shape():
    batch_size = 4
    seq_len = 60
    input_dim = 10
    hidden_dim = 64

    model = LSTMAutoencoder(
        input_dim=input_dim, hidden_dim=hidden_dim, seq_len=seq_len, num_layers=2
    )
    dummy_x = torch.randn(batch_size, seq_len, input_dim)
    reconstructed = model(dummy_x)

    assert reconstructed.shape == (batch_size, seq_len, input_dim)


def test_lstm_detector_scoring_and_threshold():
    detector = LSTMAutoencoderDetector(input_dim=10, hidden_dim=32, seq_len=60)
    X_dummy = np.random.normal(0.5, 0.05, size=(10, 60, 10)).astype(np.float32)
    scores = detector.compute_anomaly_scores(X_dummy)

    assert len(scores) == 10
    assert np.all(scores >= 0.0)

    th = detector.calibrate_threshold(X_dummy, num_std=3.0)
    assert th > detector.train_mean_error


def test_isolation_forest_detector():
    det = IsolationForestDetector(n_estimators=50)
    X_train = np.random.normal(0.5, 0.1, size=(50, 60, 10))
    det.fit(X_train)

    X_test = np.random.normal(0.5, 0.1, size=(5, 60, 10))
    scores = det.compute_anomaly_scores(X_test)
    assert len(scores) == 5
    assert det.is_fitted


def test_one_class_svm_detector():
    det = OneClassSVMDetector()
    X_train = np.random.normal(0.5, 0.1, size=(40, 60, 10))
    det.fit(X_train)

    X_test = np.random.normal(0.5, 0.1, size=(5, 60, 10))
    scores = det.compute_anomaly_scores(X_test)
    assert len(scores) == 5
    assert det.is_fitted


def test_anomaly_transformer_shape():
    model = AnomalyTransformer(input_dim=10, d_model=32, n_heads=4, seq_len=60)
    x = torch.randn(2, 60, 10)
    recon, series, prior = model(x)
    assert recon.shape == (2, 60, 10)
    assert len(series) == 2
    assert len(prior) == 2


def test_onnx_edge_inference():
    onnx_path = "models_saved/onnx/lstm_ae_fp32.onnx"
    if not os.path.exists(onnx_path):
        pytest.skip("ONNX model not exported yet.")

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_shape = sess.get_inputs()[0].shape
    dim = in_shape[2] if len(in_shape) >= 3 and isinstance(in_shape[2], int) else 10
    dummy_x = np.random.randn(1, 60, dim).astype(np.float32)
    out = sess.run(None, {sess.get_inputs()[0].name: dummy_x})
    assert out[0].shape == (1, 60, dim)
