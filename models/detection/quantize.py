"""
Model Quantization Engine for CEdge-XAI.
Applies dynamic quantization (Float32 -> Int8) to both PyTorch and ONNX models
to minimize RAM consumption and inference latency on constrained IoT edge hardware.
"""

import argparse
import logging
import os
import sys

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from models.detection.lstm_autoencoder import LSTMAutoencoder  # noqa: E402

logger = logging.getLogger(__name__)


def quantize_onnx_model(
    input_onnx_path: str = "models_saved/onnx/lstm_ae_fp32.onnx",
    output_onnx_path: str = "models_saved/quantized/lstm_ae_int8.onnx",
) -> str:
    """Apply dynamic INT8 quantization to ONNX FP32 model."""
    if not os.path.exists(input_onnx_path):
        raise FileNotFoundError(f"Input ONNX file not found: {input_onnx_path}")

    os.makedirs(os.path.dirname(output_onnx_path), exist_ok=True)
    logger.info(f"Quantizing ONNX model {input_onnx_path} to INT8: {output_onnx_path}...")

    quantize_dynamic(
        model_input=input_onnx_path,
        model_output=output_onnx_path,
        weight_type=QuantType.QInt8,
    )

    size_fp32 = os.path.getsize(input_onnx_path) / (1024 * 1024)
    size_int8 = os.path.getsize(output_onnx_path) / (1024 * 1024)
    compression = (1.0 - size_int8 / size_fp32) * 100.0

    logger.info(
        f"ONNX Quantization complete: {size_fp32:.3f} MB -> {size_int8:.3f} MB "
        f"({compression:.1f}% size reduction)"
    )

    # Validate INT8 ONNX Session
    session = ort.InferenceSession(output_onnx_path, providers=["CPUExecutionProvider"])
    dummy_input = np.random.randn(1, 60, 10).astype(np.float32)
    ort_inputs = {session.get_inputs()[0].name: dummy_input}
    _ = session.run(None, ort_inputs)
    logger.info("ONNX INT8 model validated successfully with onnxruntime.")
    return output_onnx_path


def quantize_pytorch_model(
    checkpoint_path: str = "models_saved/checkpoints/lstm_ae_best.pt",
    output_pt_path: str = "models_saved/quantized/lstm_ae_int8.pt",
    config_path: str = "config/config.yaml",
) -> str:
    """Apply dynamic quantization to PyTorch LSTM-Autoencoder."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model = LSTMAutoencoder(
        input_dim=config["models"]["lstm_ae"]["input_dim"],
        hidden_dim=config["models"]["lstm_ae"]["hidden_dim"],
        seq_len=config["preprocessing"]["window_size"],
        num_layers=config["models"]["lstm_ae"]["num_layers"],
        dropout=config["models"]["lstm_ae"]["dropout"],
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    logger.info("Quantizing PyTorch model using torch.quantization.quantize_dynamic...")
    try:
        quantized_model = torch.quantization.quantize_dynamic(
            model, {nn.LSTM, nn.Linear}, dtype=torch.qint8
        )

        os.makedirs(os.path.dirname(output_pt_path), exist_ok=True)
        torch.save(
            {
                "model_state_dict": quantized_model.state_dict(),
                "threshold": ckpt.get("threshold", 0.0),
            },
            output_pt_path,
        )

        size_fp32 = os.path.getsize(checkpoint_path) / (1024 * 1024)
        size_int8 = os.path.getsize(output_pt_path) / (1024 * 1024)
        logger.info(f"PyTorch Quantization complete: {size_fp32:.3f} MB -> {size_int8:.3f} MB")
        return output_pt_path
    except Exception as e:
        logger.warning(
            f"PyTorch eager dynamic quantization unsupported on platform: {e}. "
            "Edge deployment uses ONNX INT8 runtime."
        )
        return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize LSTM-Autoencoder models.")
    parser.add_argument("--onnx-in", type=str, default="models_saved/onnx/lstm_ae_fp32.onnx")
    parser.add_argument("--onnx-out", type=str, default="models_saved/quantized/lstm_ae_int8.onnx")
    parser.add_argument("--pt-in", type=str, default="models_saved/checkpoints/lstm_ae_best.pt")
    parser.add_argument("--pt-out", type=str, default="models_saved/quantized/lstm_ae_int8.pt")
    args = parser.parse_args()

    if os.path.exists(args.onnx_in):
        quantize_onnx_model(args.onnx_in, args.onnx_out)
    if os.path.exists(args.pt_in):
        quantize_pytorch_model(args.pt_in, args.pt_out)
