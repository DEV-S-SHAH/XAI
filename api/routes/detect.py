"""
Detection Endpoint for CEdge-XAI FastAPI Backend.
Performs real-time window anomaly detection using optimized model / ONNX runtime.
"""

from datetime import datetime
import os
import sys
import time
from fastapi import APIRouter, HTTPException
import joblib
import numpy as np
import onnxruntime as ort
import torch
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from api.schemas import DetectRequest, DetectResponse  # noqa: E402
from models.detection.lstm_autoencoder import LSTMAutoencoder  # noqa: E402

router = APIRouter()

# Global cached detector state
_MODEL = None
_ORT_SESSION = None
_SCALER = None
_THRESHOLD = 0.0125
_CONFIG = None


def get_detector_resources():
    global _MODEL, _ORT_SESSION, _SCALER, _THRESHOLD, _CONFIG
    if _CONFIG is None:
        cfg_path = "config/config.yaml"
        if os.path.exists(cfg_path):
            with open(cfg_path, "r") as f:
                _CONFIG = yaml.safe_load(f)

    if _SCALER is None and os.path.exists("models_saved/checkpoints/scaler.joblib"):
        _SCALER = joblib.load("models_saved/checkpoints/scaler.joblib")

    # Try ONNX first for fastest edge inference
    onnx_path = "models_saved/quantized/lstm_ae_int8.onnx"
    if not os.path.exists(onnx_path):
        onnx_path = "models_saved/onnx/lstm_ae_fp32.onnx"

    if _ORT_SESSION is None and os.path.exists(onnx_path):
        try:
            _ORT_SESSION = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        except Exception:
            _ORT_SESSION = None

    # Fallback to PyTorch
    ckpt_path = "models_saved/checkpoints/lstm_ae_best.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = "models_saved/checkpoints/lstm_ae.pt"

    if _MODEL is None and os.path.exists(ckpt_path):
        input_dim = _CONFIG["models"]["lstm_ae"]["input_dim"] if _CONFIG else 10
        hidden_dim = _CONFIG["models"]["lstm_ae"]["hidden_dim"] if _CONFIG else 64
        seq_len = _CONFIG["preprocessing"]["window_size"] if _CONFIG else 60
        _MODEL = LSTMAutoencoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            seq_len=seq_len,
            num_layers=2,
            dropout=0.0,
        )
        ckpt_data = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt_data["model_state_dict"] if "model_state_dict" in ckpt_data else ckpt_data
        _MODEL.load_state_dict(state_dict)
        _MODEL.eval()

    # Load threshold
    meta_path = "models_saved/checkpoints/lstm_ae_meta.yaml"
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = yaml.safe_load(f)
            _THRESHOLD = meta.get("threshold", 0.0125)

    return _MODEL, _ORT_SESSION, _SCALER, _THRESHOLD


@router.post("/detect", response_model=DetectResponse)
async def detect_anomaly(req: DetectRequest):
    model, ort_session, scaler, threshold = get_detector_resources()
    expected_dim = len(_CONFIG["data"]["features"]) if _CONFIG else 10

    t_start = time.perf_counter()

    if req.window is not None:
        window_arr = np.array(req.window, dtype=np.float32)
        if window_arr.ndim != 2 or window_arr.shape[0] != 60 or window_arr.shape[1] != expected_dim:
            raise HTTPException(
                status_code=400,
                detail=f"Window shape must be (60, {expected_dim}), got {window_arr.shape}",
            )
    elif req.current_reading is not None:
        # Replicate reading across 60 timesteps as a synthetic steady window
        row = list(req.current_reading.dict().values())
        if scaler:
            row = scaler.transform([row])[0]
        window_arr = np.tile(row, (60, 1)).astype(np.float32)
    else:
        raise HTTPException(
            status_code=400,
            detail="Either 'window' or 'current_reading' must be provided in request.",
        )

    # Add batch dimension: (1, 60, 13)
    batch_input = np.expand_dims(window_arr, axis=0)

    # Perform inference
    if ort_session is not None:
        in_name = ort_session.get_inputs()[0].name
        reconstruction = ort_session.run(None, {in_name: batch_input})[0]
    elif model is not None:
        with torch.no_grad():
            tensor_in = torch.tensor(batch_input, dtype=torch.float32)
            reconstruction = model(tensor_in).numpy()
    else:
        raise HTTPException(
            status_code=503,
            detail="Detector model checkpoint not yet loaded or trained.",
        )

    # Compute composite reconstruction anomaly score
    window_mse = float(np.mean((reconstruction - batch_input) ** 2))
    latest_mse = float(np.mean((reconstruction[0, -1, :] - batch_input[0, -1, :]) ** 2))
    anomaly_score = 0.5 * window_mse + 0.5 * latest_mse

    is_anomaly = bool(anomaly_score > threshold)
    ratio = anomaly_score / (threshold + 1e-6)
    confidence = float(min(1.0, max(0.5, 0.5 + 0.5 * abs(ratio - 1.0) / (ratio + 1.0))))

    t_end = time.perf_counter()
    inference_time_ms = (t_end - t_start) * 1000.0

    ts = req.timestamp or datetime.utcnow().isoformat()

    return DetectResponse(
        timestamp=ts,
        anomaly=is_anomaly,
        anomaly_score=round(anomaly_score, 6),
        threshold=round(threshold, 6),
        confidence=round(confidence, 4),
        inference_time_ms=round(inference_time_ms, 3),
    )
