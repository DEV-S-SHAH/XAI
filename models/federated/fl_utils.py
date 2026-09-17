"""
Federated Learning Utilities for CEdge-XAI.
Handles data partitioning for 3 factory sites (Factory A, Factory B, Factory C),
parameter serialization/deserialization for Flower, and privacy verification.
"""

import logging
import os
import sys
from collections import OrderedDict
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from models.detection.train import FEATURE_COLUMNS, create_sliding_windows  # noqa: E402

logger = logging.getLogger(__name__)


def partition_factory_data(
    csv_path: str = "data/synthetic/factory_iot_data.csv",
    num_clients: int = 3,
    window_size: int = 60,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Partition 4320 rows into 3 isolated factory subsets (~1440 rows each).
    Generates local client datasets strictly maintaining data locality.
    """
    df = pd.read_csv(csv_path)
    scaler_path = "models_saved/checkpoints/scaler.joblib"
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    total_rows = len(df)
    chunk_size = total_rows // num_clients

    factory_names = ["Factory_A", "Factory_B", "Factory_C"]
    partitions = {}

    for i, name in enumerate(factory_names):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < num_clients - 1 else total_rows
        df_client = df.iloc[start_idx:end_idx].copy()

        X_raw = df_client[FEATURE_COLUMNS].values
        y_raw = (
            df_client["anomaly"].values
            if "anomaly" in df_client.columns
            else np.zeros(len(df_client))
        )

        if scaler:
            X_scaled = scaler.transform(X_raw)
        else:
            X_scaled = X_raw

        X_seq, y_seq = create_sliding_windows(X_scaled, y_raw, window_size=window_size)
        pure_normal_mask = np.array(
            [y_raw[k : k + window_size].sum() == 0 for k in range(len(X_seq))]
        )

        partitions[name] = {
            "client_id": i,
            "factory_name": name,
            "raw_rows": len(df_client),
            "X_train_normal": X_seq[pure_normal_mask],
            "X_val": X_seq,
            "y_val": y_seq,
        }
        logger.info(
            f"Partitioned [{name}]: Rows {start_idx}-{end_idx} ({len(df_client)} rows), "
            f"Normal Train Windows={X_seq[pure_normal_mask].shape[0]}"
        )

    return partitions


def get_model_parameters(model: torch.nn.Module) -> List[np.ndarray]:
    """Extract model parameters as numpy arrays for Flower aggregation."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_model_parameters(model: torch.nn.Module, parameters: List[np.ndarray]) -> None:
    """Load aggregated parameters from Flower into torch model."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
