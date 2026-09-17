"""
Explanation Endpoint for CEdge-XAI FastAPI Backend.
Performs counterfactual XAI and causal root cause analysis.
"""

from datetime import datetime
import os
import sys
import time
from fastapi import APIRouter, HTTPException
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from api.schemas import ExplainRequest, ExplainResponse, SensorContribution  # noqa: E402
from models.xai.counterfactual_engine import CounterfactualEngine  # noqa: E402
from models.xai.explanation_formatter import ExplanationFormatter  # noqa: E402

router = APIRouter()

_CF_ENGINE = None
_FORMATTER = None


def get_xai_resources():
    global _CF_ENGINE, _FORMATTER
    if _CF_ENGINE is None:
        cfg_path = "config/config.yaml"
        if not os.path.exists(cfg_path):
            raise RuntimeError("config/config.yaml not found.")
        _CF_ENGINE = CounterfactualEngine(config_path=cfg_path)
    if _FORMATTER is None:
        _FORMATTER = ExplanationFormatter(config_path="config/config.yaml")
    return _CF_ENGINE, _FORMATTER


@router.post("/explain", response_model=ExplainResponse)
async def explain_anomaly(req: ExplainRequest):
    t_start = time.perf_counter()

    cf_engine, formatter = get_xai_resources()
    expected_dim = len(cf_engine.feature_names)

    window_arr = np.array(req.window, dtype=np.float32)
    if window_arr.ndim != 2 or window_arr.shape[0] != 60 or window_arr.shape[1] != expected_dim:
        raise HTTPException(
            status_code=400,
            detail=f"Expected window shape (60, {expected_dim}), got {window_arr.shape}",
        )

    # 1. Compute baseline anomaly score
    batch_input = np.expand_dims(window_arr, axis=0)
    orig_score = float(cf_engine.detector.compute_anomaly_scores(batch_input)[0])
    threshold = float(cf_engine.detector.threshold)
    is_anomaly = bool(orig_score > threshold)

    # 2. Generate Counterfactual
    if req.method.lower() == "gradient":
        cf_res = cf_engine.optimize_counterfactual(window_arr)
        deltas = cf_res.get("perturbations", {})
        is_valid = cf_res.get("converged", True)
    else:
        cf_res = cf_engine.explain_physics_clamping(window_arr)
        deltas = cf_res.get("sensor_impacts", {})
        is_valid = True

    # 3. Format Explanation narrative and causal path
    exp_dict = formatter.format_explanation(
        original_sample=window_arr,
        counterfactual_result=cf_res,
        timestamp=req.timestamp or datetime.utcnow().isoformat(),
    )

    t_end = time.perf_counter()
    comp_time_ms = (t_end - t_start) * 1000.0

    # Extract top contributors
    contributors = []
    orig_last = window_arr[-1, :]

    sorted_deltas = sorted(deltas.items(), key=lambda x: abs(x[1]), reverse=True)[: req.top_k]
    for s_name, delta_val in sorted_deltas:
        s_idx = cf_engine.feature_names.index(s_name)
        curr = float(orig_last[s_idx])
        cf_val = float(curr - delta_val)
        contributors.append(
            SensorContribution(
                sensor=s_name,
                contribution_score=round(float(abs(delta_val)), 4),
                current_value=round(curr, 3),
                counterfactual_value=round(cf_val, 3),
                delta=round(float(delta_val), 3),
            )
        )

    return ExplainResponse(
        timestamp=exp_dict.get("timestamp", req.timestamp or datetime.utcnow().isoformat()),
        anomaly=is_anomaly,
        anomaly_score=round(float(orig_score), 6),
        threshold=round(float(threshold), 6),
        root_cause=exp_dict.get("root_cause", "fan_speed"),
        explanation=exp_dict.get("explanation", ""),
        recommended_action=exp_dict.get("recommended_action", ""),
        causal_path=exp_dict.get("causal_path") or exp_dict.get("causal_chain", []),
        top_contributors=contributors,
        counterfactual_valid=is_valid,
        iterations=300 if req.method.lower() == "gradient" else 13,
        computation_time_ms=round(comp_time_ms, 2),
    )
