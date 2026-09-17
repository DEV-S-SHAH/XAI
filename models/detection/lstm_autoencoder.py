"""
LSTM-Autoencoder for IoT Time Series Anomaly Detection (Primary Model).
Features 2-layer Encoder and 2-layer Decoder with reconstruction-based anomaly scoring.
Fully differentiable for gradient-based counterfactual generation and exportable to ONNX.
"""

import logging
import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class LSTMEncoder(nn.Module):
    """2-layer LSTM Encoder."""

    def __init__(
        self, input_dim: int = 10, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim, self.hidden_dim, self.num_layers = input_dim, hidden_dim, num_layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, (h_n, c_n) = self.lstm(x)
        return h_n, c_n


class LSTMDecoder(nn.Module):
    """2-layer LSTM Decoder."""

    def __init__(
        self,
        output_dim: int = 10,
        hidden_dim: int = 64,
        seq_len: int = 60,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.output_dim, self.hidden_dim, self.seq_len, self.num_layers = (
            output_dim,
            hidden_dim,
            seq_len,
            num_layers,
        )
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, h_n: torch.Tensor, c_n: torch.Tensor) -> torch.Tensor:
        context = h_n[-1].unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.lstm(context, (h_n, c_n))
        return self.fc(out)


class LSTMAutoencoder(nn.Module):
    """Complete LSTM-Autoencoder network."""

    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 64,
        seq_len: int = 60,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim, self.hidden_dim, self.seq_len, self.num_layers, self.dropout = (
            input_dim,
            hidden_dim,
            seq_len,
            num_layers,
            dropout,
        )
        self.encoder = LSTMEncoder(
            input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout
        )
        self.decoder = LSTMDecoder(
            output_dim=input_dim,
            hidden_dim=hidden_dim,
            seq_len=seq_len,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            Reconstructed tensor of shape (batch_size, seq_len, input_dim)
        """
        h_n, c_n = self.encoder(x)
        reconstruction = self.decoder(h_n, c_n)
        return reconstruction


class LSTMAutoencoderDetector:
    """Production wrapper for LSTM-Autoencoder with calibration and inference utilities."""

    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 64,
        seq_len: int = 60,
        num_layers: int = 2,
        dropout: float = 0.1,
        lr: float = 0.001,
        device: Optional[str] = None,
        checkpoint_path: str = "models/checkpoints/lstm_ae_best.pt",
    ):
        self.input_dim, self.hidden_dim, self.seq_len = input_dim, hidden_dim, seq_len
        self.num_layers, self.dropout, self.lr = num_layers, dropout, lr
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = LSTMAutoencoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            seq_len=self.seq_len,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)
        self.criterion = nn.MSELoss(reduction="none")
        self.threshold = self.train_mean_error = self.train_std_error = 0.0
        self.is_fitted = False

    def fit(
        self,
        X_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32,
        patience: int = 10,
    ) -> "LSTMAutoencoderDetector":
        """
        Train LSTM-Autoencoder on normal sequences only.
        """
        self.model.train()
        train_tensor = torch.tensor(X_train, dtype=torch.float32)
        dataset = TensorDataset(train_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        best_loss = float("inf")
        patience_counter = 0

        logger.info(
            f"Training LSTM-Autoencoder on {len(X_train)} normal sequences for {epochs} epochs on {self.device}..."
        )

        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            self.model.train()
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()
                reconstructed = self.model(batch_x)
                loss = nn.MSELoss()(reconstructed, batch_x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch_x)

            epoch_loss = total_loss / len(X_train)

            # Validation loss if provided
            val_loss = epoch_loss
            if X_val is not None:
                self.model.eval()
                val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
                with torch.no_grad():
                    val_recon = self.model(val_tensor)
                    val_loss = nn.MSELoss()(val_recon, val_tensor).item()

            scheduler.step(val_loss)

            if epoch % 5 == 0 or epoch == 1:
                logger.info(
                    f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {epoch_loss:.6f} | Val Loss: {val_loss:.6f}"
                )

            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                self.save()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break

        # Load best model and calibrate threshold on normal training set
        self.load()
        self.calibrate_threshold(X_train)
        self.is_fitted = True
        return self

    def calibrate_threshold(self, X_normal: np.ndarray, num_std: float = 3.0) -> float:
        """Calibrate statistical threshold: mean + 3*std of normal reconstruction errors."""
        self.model.eval()
        scores = self.compute_anomaly_scores(X_normal)
        self.train_mean_error = float(np.mean(scores))
        self.train_std_error = float(np.std(scores))
        self.threshold = float(self.train_mean_error + num_std * self.train_std_error)
        logger.info(
            f"Calibrated 3-Sigma Threshold: {self.threshold:.6f} "
            f"(Mean: {self.train_mean_error:.6f}, Std: {self.train_std_error:.6f})"
        )
        return self.threshold

    def compute_anomaly_scores(self, X: np.ndarray, per_feature: bool = False) -> np.ndarray:
        """
        Compute reconstruction error for each sequence window.
        Args:
            X: Input sequence array of shape (N, seq_len, input_dim)
            per_feature: If True, returns error per feature of shape (N, input_dim)
        Returns:
            MSE reconstruction scores
        """
        self.model.eval()
        tensor_x = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            reconstructed = self.model(tensor_x)
            diff = (reconstructed - tensor_x) ** 2
            if per_feature:
                # Average over sequence length: (N, input_dim)
                return diff.mean(dim=1).cpu().numpy()
            # Combine window-level reconstruction error and latest-timestep error
            # This captures temporal context while ensuring instantaneous sensitivity to new anomalies
            window_err = diff.mean(dim=(1, 2))
            latest_err = diff[:, -1, :].mean(dim=-1)
            score = 0.5 * window_err + 0.5 * latest_err
            return score.cpu().numpy()

    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """Predict binary anomaly flags (1=anomaly, 0=normal)."""
        thresh = threshold if threshold is not None else self.threshold
        scores = self.compute_anomaly_scores(X)
        return (scores > thresh).astype(int)

    def save(self, path: Optional[str] = None) -> str:
        """Save model state dict and threshold."""
        target = path or self.checkpoint_path
        os.makedirs(os.path.dirname(target), exist_ok=True)
        ckpt = {
            "model_state_dict": self.model.state_dict(),
            "threshold": self.threshold,
            "train_mean_error": self.train_mean_error,
            "train_std_error": self.train_std_error,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "seq_len": self.seq_len,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }
        torch.save(ckpt, target)
        return target

    def load(self, path: Optional[str] = None) -> "LSTMAutoencoderDetector":
        """Load model state dict and parameters."""
        target = path or self.checkpoint_path
        if not os.path.exists(target):
            raise FileNotFoundError(f"Checkpoint not found at {target}")
        ckpt = torch.load(target, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.threshold = ckpt.get("threshold", 0.0)
        self.train_mean_error = ckpt.get("train_mean_error", 0.0)
        self.train_std_error = ckpt.get("train_std_error", 0.0)
        self.is_fitted = True
        return self
