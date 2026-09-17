"""
FastAPI Main Application for XAI for IOT Anomaly Detection.
Provides RESTful APIs for real-time edge inference, counterfactual explainability, and causal root cause diagnosis.
"""

import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from api.routes.detect import router as detect_router  # noqa: E402
from api.routes.explain import router as explain_router  # noqa: E402
from api.schemas import ModelInfoResponse  # noqa: E402

app = FastAPI(
    title="XAI for IOT Anomaly Detection",
    description="Real-Time Anomaly Detection, Counterfactual Explanations, and Causal Discovery for Cyber-Physical Systems.",
    version="1.0.0",
)

# Enable CORS for Streamlit frontend and distributed clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detect_router, prefix="/api/v1", tags=["Detection"])
app.include_router(explain_router, prefix="/api/v1", tags=["Explainability"])


@app.get("/api/v1/health", tags=["System"])
async def health_check():
    """Health check endpoint indicating microservice availability and active edge runtime."""
    onnx_exists = os.path.exists("models_saved/quantized/lstm_ae_int8.onnx") or os.path.exists(
        "models_saved/onnx/lstm_ae_fp32.onnx"
    )
    pt_exists = os.path.exists("models_saved/checkpoints/lstm_ae_best.pt") or os.path.exists(
        "models_saved/checkpoints/lstm_ae.pt"
    )

    return {
        "status": "healthy" if (onnx_exists or pt_exists) else "degraded",
        "version": "1.0.0",
        "active_model": "LSTM-Autoencoder",
        "edge_accelerated": onnx_exists,
        "device": "cpu",
    }


@app.get("/api/v1/model-info", response_model=ModelInfoResponse, tags=["System"])
async def model_info():
    """Returns metadata and hyperparameter specification for deployed models."""
    cfg_path = "config/config.yaml"
    cfg = {}
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)

    threshold = 0.0125
    meta_path = "models_saved/checkpoints/lstm_ae_meta.yaml"
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = yaml.safe_load(f)
            threshold = meta.get("threshold", 0.0125)

    sensor_names = cfg.get("data", {}).get(
        "features",
        [
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
        ],
    )

    return ModelInfoResponse(
        model_name="XAI for IOT Anomaly Detection LSTM-Autoencoder",
        architecture="2-Layer Encoder-Decoder LSTM",
        num_sensors=len(sensor_names),
        sensor_names=sensor_names,
        window_size=cfg.get("preprocessing", {}).get("window_size", 60),
        threshold=round(float(threshold), 6),
        onnx_deployed=os.path.exists("models_saved/onnx/lstm_ae_fp32.onnx"),
        int8_quantized=os.path.exists("models_saved/quantized/lstm_ae_int8.onnx"),
        version="1.0.0",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
