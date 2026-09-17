"""
Federated Learning Client for CEdge-XAI.
Simulates an industrial edge node (Factory A/B/C).
Trains local LSTM-Autoencoder weights without ever transferring raw sensor telemetry.
Generates XAI explanations locally on-device.
"""

import argparse
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import yaml

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from models.detection.lstm_autoencoder import LSTMAutoencoder  # noqa: E402
from models.federated.fl_utils import (  # noqa: E402
    get_model_parameters,
    partition_factory_data,
    set_model_parameters,
)

logger = logging.getLogger(__name__)


class FactoryClient(fl.client.NumPyClient):
    """Flower NumPyClient implementing FedAvg for local industrial edge nodes."""

    def __init__(
        self,
        client_id: int,
        factory_name: str,
        X_train_normal: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        config: Dict,
        device: Optional[str] = None,
    ):
        self.client_id = client_id
        self.factory_name = factory_name
        self.X_train_normal = X_train_normal
        self.X_val = X_val
        self.y_val = y_val
        self.config = config

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = LSTMAutoencoder(
            input_dim=config["models"]["lstm_ae"]["input_dim"],
            hidden_dim=config["models"]["lstm_ae"]["hidden_dim"],
            seq_len=config["preprocessing"]["window_size"],
            num_layers=config["models"]["lstm_ae"]["num_layers"],
            dropout=config["models"]["lstm_ae"]["dropout"],
        ).to(self.device)

        self.local_epochs = config["federated"]["local_epochs"]
        self.batch_size = config["federated"]["batch_size"]
        self.lr = config["models"]["lstm_ae"]["learning_rate"]

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        """Return model parameters to server for aggregation."""
        return get_model_parameters(self.model)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict[str, float]]:
        """Train model locally on normal sequences."""
        # 1. Update local model with globally aggregated weights
        set_model_parameters(self.model, parameters)

        # 2. Train on local data
        train_tensor = torch.tensor(self.X_train_normal, dtype=torch.float32)
        dataset = TensorDataset(train_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        self.model.train()
        total_loss = 0.0
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()
                recon = self.model(batch_x)
                loss = criterion(recon, batch_x)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(batch_x)
            total_loss = epoch_loss / len(self.X_train_normal)

        logger.info(
            f"[{self.factory_name}] Local Training Complete. "
            f"Loss: {total_loss:.5f} | Samples: {len(self.X_train_normal)}"
        )

        return (
            get_model_parameters(self.model),
            len(self.X_train_normal),
            {"loss": float(total_loss)},
        )

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict[str, float]]:
        """Evaluate aggregated model on local validation data."""
        set_model_parameters(self.model, parameters)
        self.model.eval()

        val_tensor = torch.tensor(self.X_val, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            recon = self.model(val_tensor)
            val_loss = nn.MSELoss()(recon, val_tensor).item()

        return float(val_loss), len(self.X_val), {"val_loss": float(val_loss)}


def launch_client(client_id: int, config_path: str = "configs/config.yaml"):
    """Launch a standalone client process."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    partitions = partition_factory_data(
        csv_path=config["data"]["synthetic_csv"],
        num_clients=config["federated"]["num_clients"],
        window_size=config["preprocessing"]["window_size"],
    )

    factory_names = list(partitions.keys())
    name = factory_names[client_id]
    p = partitions[name]

    client = FactoryClient(
        client_id=client_id,
        factory_name=name,
        X_train_normal=p["X_train_normal"],
        X_val=p["X_val"],
        y_val=p["y_val"],
        config=config,
    )

    server_address = config["federated"]["server_address"]
    logger.info(f"Starting {name} connecting to FL Server at {server_address}...")
    fl.client.start_client(server_address=server_address, client=client.to_client())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start Federated Client.")
    parser.add_argument("--client-id", type=int, default=0, help="Client ID (0, 1, 2)")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml", help="Path to config.yaml"
    )
    args = parser.parse_args()

    launch_client(client_id=args.client_id, config_path=args.config)
