"""
Model Quantization Engine for XAI for IOT Anomaly Detection.
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
    """Apply dynamic INT8 quantization to ONNX FP32 model including LSTM layers."""
    if not os.path.exists(input_onnx_path):
        raise FileNotFoundError(f"Input ONNX file not found: {input_onnx_path}")

    os.makedirs(os.path.dirname(output_onnx_path), exist_ok=True)
    logger.info(f"Quantizing ONNX model {input_onnx_path} to INT8: {output_onnx_path}...")

    import onnx
    from onnx import numpy_helper

    # Fold constant slice/concat/unsqueeze operations so LSTM weight tensors become direct initializers
    model = onnx.load(input_onnx_path)
    init_map = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}

    changed = True
    while changed:
        changed = False
        new_nodes = []
        for node in model.graph.node:
            if node.op_type in ["Slice", "Concat", "Unsqueeze", "Reshape", "Transpose"] and all(
                inp in init_map for inp in node.input if inp != ""
            ):
                sub_inputs = [
                    onnx.helper.make_tensor_value_info(
                        i,
                        onnx.TensorProto.FLOAT if init_map[i].dtype == np.float32 else onnx.TensorProto.INT64,
                        list(init_map[i].shape),
                    )
                    for i in node.input
                    if i != ""
                ]
                sub_outputs = [
                    onnx.helper.make_tensor_value_info(
                        o,
                        onnx.TensorProto.FLOAT if any(init_map[i].dtype == np.float32 for i in node.input if i != "") else onnx.TensorProto.INT64,
                        None,
                    )
                    for o in node.output
                ]
                sub_graph = onnx.helper.make_graph(
                    [node],
                    "sub",
                    sub_inputs,
                    sub_outputs,
                    [numpy_helper.from_array(init_map[i], name=i) for i in node.input if i != ""],
                )
                sub_model = onnx.helper.make_model(sub_graph, opset_imports=model.opset_import)
                sess = ort.InferenceSession(sub_model.SerializeToString())
                feed = {i: init_map[i] for i in node.input if i != ""}
                res = sess.run(None, feed)
                for out_name, val in zip(node.output, res):
                    init_map[out_name] = val
                    model.graph.initializer.append(numpy_helper.from_array(val, name=out_name))
                changed = True
            else:
                new_nodes.append(node)
        model.graph.ClearField("node")
        model.graph.node.extend(new_nodes)

    used_inputs = set()
    for node in model.graph.node:
        for i in node.input:
            used_inputs.add(i)
    model.graph.ClearField("initializer")
    for name, arr in init_map.items():
        if name in used_inputs:
            model.graph.initializer.append(numpy_helper.from_array(arr, name=name))

    folded_temp_path = output_onnx_path + ".folded.onnx"
    onnx.save(model, folded_temp_path)

    quantize_dynamic(
        model_input=folded_temp_path,
        model_output=output_onnx_path,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["LSTM", "MatMul", "Gemm"],
    )

    if os.path.exists(folded_temp_path):
        os.remove(folded_temp_path)

    size_fp32 = os.path.getsize(input_onnx_path) / (1024 * 1024)
    size_int8 = os.path.getsize(output_onnx_path) / (1024 * 1024)
    compression = (1.0 - size_int8 / size_fp32) * 100.0

    logger.info(
        f"ONNX Quantization complete: {size_fp32:.3f} MB -> {size_int8:.3f} MB "
        f"({compression:.1f}% size reduction, {size_fp32/size_int8:.2f}x compression)"
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
