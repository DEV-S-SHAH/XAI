"""
ONNX Export Engine for LSTM-Autoencoder.
Exports the PyTorch LSTM-Autoencoder checkpoint to ONNX FP32 with dynamic batch axes
and validates inference output equivalence with onnxruntime.
"""

import argparse
import logging
import os
import sys

import numpy as np
import onnxruntime as ort
import torch
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from models.detection.lstm_autoencoder import LSTMAutoencoder  # noqa: E402

logger = logging.getLogger(__name__)


def export_lstm_ae_to_onnx(
    checkpoint_path: str = "models_saved/checkpoints/lstm_ae_best.pt",
    output_onnx_path: str = "models_saved/onnx/lstm_ae_fp32.onnx",
    config_path: str = "config/config.yaml",
) -> str:
    """Export trained PyTorch LSTM-Autoencoder to ONNX FP32 format."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    input_dim = config["models"]["lstm_ae"]["input_dim"]
    hidden_dim = config["models"]["lstm_ae"]["hidden_dim"]
    seq_len = config["preprocessing"]["window_size"]
    num_layers = config["models"]["lstm_ae"]["num_layers"]
    dropout = config["models"]["lstm_ae"]["dropout"]

    logger.info(f"Loading PyTorch checkpoint from {checkpoint_path}...")
    model = LSTMAutoencoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        seq_len=seq_len,
        num_layers=num_layers,
        dropout=dropout,
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Create dummy input: (batch_size=1, seq_len=60, input_dim=10)
    dummy_input = torch.randn(1, seq_len, input_dim, dtype=torch.float32)

    os.makedirs(os.path.dirname(output_onnx_path), exist_ok=True)
    logger.info(f"Exporting model to ONNX: {output_onnx_path}...")

    torch.onnx.export(
        model,
        dummy_input,
        output_onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["reconstruction"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "reconstruction": {0: "batch_size"},
        },
    )

    # Ensure single self-contained ONNX file without external .data file
    data_file = f"{output_onnx_path}.data"
    if os.path.exists(data_file):
        import onnx

        onnx_model = onnx.load(output_onnx_path, load_external_data=True)
        onnx.save_model(onnx_model, output_onnx_path, save_as_external_data=False)
        if os.path.exists(data_file):
            os.remove(data_file)

    # Validate with onnxruntime
    session = ort.InferenceSession(output_onnx_path, providers=["CPUExecutionProvider"])
    ort_inputs = {session.get_inputs()[0].name: dummy_input.numpy()}
    ort_outs = session.run(None, ort_inputs)

    with torch.no_grad():
        torch_outs = model(dummy_input).numpy()

    diff = np.max(np.abs(torch_outs - ort_outs[0]))
    logger.info(f"ONNX export successful. Max numerical discrepancy: {diff:.6e}")
    file_size_mb = os.path.getsize(output_onnx_path) / (1024 * 1024)
    logger.info(f"Exported ONNX FP32 Model Size: {file_size_mb:.3f} MB")

    return output_onnx_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch LSTM-AE to ONNX FP32.")
    parser.add_argument(
        "--checkpoint", type=str, default="models_saved/checkpoints/lstm_ae_best.pt"
    )
    parser.add_argument("--output", type=str, default="models_saved/onnx/lstm_ae_fp32.onnx")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    export_lstm_ae_to_onnx(args.checkpoint, args.output, args.config)
