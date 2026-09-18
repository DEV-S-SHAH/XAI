"""
Unit and Integration Tests for Novelty 1: Actionable What-If Explanations.
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.xai.actionable_what_if import ActionableWhatIfExplainer


def test_actionable_what_if_explainer_init():
    explainer = ActionableWhatIfExplainer(config_path="config/config.yaml")
    assert len(explainer.feature_names) == 10
    assert explainer.normal_medians_scaled.shape[0] == 10
    assert explainer.detector is not None


def test_actionable_counterfactual_generation():
    explainer = ActionableWhatIfExplainer(config_path="config/config.yaml")
    # Synthetic anomalous window: fan speed dropped to 0.0
    dummy_win = np.full((60, 10), 0.5, dtype=np.float32)
    fan_idx = explainer.feature_names.index("fan_speed")
    dummy_win[:, fan_idx] = 0.0  # Anomaly injection

    res = explainer.generate_counterfactual(dummy_win)

    # Assert required metrics are present and valid
    assert "validity" in res
    assert "sparsity" in res
    assert "proximity_l1" in res
    assert "proximity_l2" in res
    assert "actionability" in res
    assert "latency_ms" in res
    assert "score_reduction" in res
    assert "score_reduction_pct" in res
    assert "explanation_narrative" in res

    # Check validity and physical constraints
    assert res["validity"] is True
    assert res["counterfactual_score"] <= res["threshold"]
    assert res["score_reduction"] > 0
    assert res["sparsity"] >= 1
    assert res["latency_ms"] < 200.0  # Must be fast (< 200 ms)
    assert "WHAT-IF ACTION" in res["explanation_narrative"]


def test_baseline_comparisons():
    explainer = ActionableWhatIfExplainer(config_path="config/config.yaml")
    dummy_win = np.full((60, 10), 0.5, dtype=np.float32)
    fan_idx = explainer.feature_names.index("fan_speed")
    dummy_win[:, fan_idx] = 0.0

    b1_res = explainer.generate_baseline_ablation(dummy_win)
    assert "root_cause" in b1_res
    assert "score_reduction" in b1_res
    assert b1_res["sparsity"] == 1

    b2_res = explainer.generate_baseline_gradient(dummy_win, steps=20)
    assert "counterfactual_score" in b2_res
    assert b2_res["sparsity"] >= 1
