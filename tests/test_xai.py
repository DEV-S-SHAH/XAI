"""
Unit Tests for Explainable AI Engines in CEdge-XAI.
Tests Physics Clamping, Gradient Optimization, Causal Graph Traversal, and Narrative Formatting.
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.xai.causal_graph import CausalDiscoveryEngine  # noqa: E402
from models.xai.counterfactual_engine import CounterfactualEngine  # noqa: E402
from models.xai.explanation_formatter import ExplanationFormatter  # noqa: E402


def test_counterfactual_clamping_method():
    cfg_path = "config/config.yaml"
    if not os.path.exists(cfg_path):
        pytest.skip("config.yaml missing")

    engine = CounterfactualEngine(config_path=cfg_path)
    n_feats = len(engine.feature_names)
    dummy_window = np.random.normal(0.5, 0.1, size=(60, n_feats)).astype(np.float32)
    # Spike fan_speed to 0.0
    fan_idx = engine.feature_names.index("fan_speed") if "fan_speed" in engine.feature_names else 0
    dummy_window[:, fan_idx] = 0.0

    res = engine.explain_physics_clamping(dummy_window)
    assert "root_cause" in res
    assert "impact_score" in res
    assert "explanation" in res
    assert len(res["sensor_impacts"]) == n_feats


def test_counterfactual_gradient_optimization():
    cfg_path = "config/config.yaml"
    if not os.path.exists(cfg_path):
        pytest.skip("config.yaml missing")

    engine = CounterfactualEngine(config_path=cfg_path)
    n_feats = len(engine.feature_names)
    dummy_window = np.random.normal(0.5, 0.1, size=(60, n_feats)).astype(np.float32)
    fan_idx = engine.feature_names.index("fan_speed") if "fan_speed" in engine.feature_names else 0
    dummy_window[:, fan_idx] = 0.0

    res = engine.optimize_counterfactual(dummy_window, steps=20)
    assert "top_perturbed_sensor" in res
    assert "perturbations" in res
    assert len(res["perturbations"]) == n_feats


def test_causal_graph_traversal():
    engine = CausalDiscoveryEngine(config_path="config/config.yaml")
    engine.run_discovery()

    assert engine.graph.number_of_nodes() == 10
    assert engine.graph.number_of_edges() > 0

    chain = engine.get_causal_chain("fan_speed", max_depth=3)
    assert len(chain) >= 1
    assert chain[0] == "fan_speed"


def test_explanation_formatter_tables():
    t3, t4 = ExplanationFormatter.generate_paper_tables(
        save_table3="paper_assets/tables/xai_comparison.csv",
        save_table4="paper_assets/tables/counterfactual_accuracy.csv",
    )
    assert len(t3) >= 5
    assert len(t4) >= 5
    assert "Anomaly ID" in t3.columns
    assert "Predicted Root Cause" in t4.columns
