"""
Unit and Integration Tests for Novelty 2: Split-Brain Edge Architecture.
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.detection.split_brain_engine import SplitBrainEdgeEngine


def test_split_brain_engine_init():
    engine = SplitBrainEdgeEngine(config_path="config/config.yaml")
    assert engine.session is not None
    assert engine.model_size_mb < 0.30  # Quantized model must be compact (< 300 KB)


def test_edge_detection_latency_constraint():
    engine = SplitBrainEdgeEngine(config_path="config/config.yaml")
    dummy_win = np.random.randn(1, 60, 10).astype(np.float32)

    # Warmup
    for _ in range(5):
        engine.detect_edge(dummy_win)

    # Measure latency
    is_anom, score, lat_ms = engine.detect_edge(dummy_win)
    assert isinstance(is_anom, bool)
    assert isinstance(score, float)
    assert lat_ms < 5.0  # Must satisfy < 5 ms constraint!


def test_server_explanation_latency_constraint():
    engine = SplitBrainEdgeEngine(config_path="config/config.yaml")
    dummy_win = np.random.randn(1, 60, 10).astype(np.float32)

    exp, lat_ms = engine.explain_server(dummy_win)
    assert isinstance(exp, dict)
    assert lat_ms < 200.0  # Must satisfy < 200 ms constraint!


def test_split_brain_benchmark_computation():
    engine = SplitBrainEdgeEngine(config_path="config/config.yaml")
    # Normal baseline windows centered around 0.5 (scaled normal operating zone)
    dummy_windows = np.full((20, 60, 10), 0.5, dtype=np.float32)
    # Inject anomaly in window 2: spike fan speed to 0.0
    fan_idx = engine.server_explainer.feature_names.index("fan_speed")
    dummy_windows[2, :, fan_idx] = 0.0

    dummy_flags = np.zeros(20, dtype=int)
    dummy_flags[2] = 1  # 1 anomaly out of 20

    res = engine.run_benchmark_comparison(dummy_windows, dummy_flags)

    assert "baseline" in res
    assert "proposed" in res
    assert "savings" in res
    # Proposed should avoid XAI executions on normal windows
    assert res["proposed"]["avoided_xai_executions"] > 0
    assert res["savings"]["bandwidth_reduction_pct"] > 50.0
