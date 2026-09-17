"""
Anomaly Transformer with Association Discrepancy for Time Series Anomaly Detection.
Based on Xu et al. (ICLR 2022): "Anomaly Transformer: Time Series Anomaly Detection
with Association Discrepancy". Simplified 2-layer architecture.
"""

import logging
import math
import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class PositionalEmbedding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, d_model)
        return self.pe[:, : x.size(1)]


class AnomalyAttention(nn.Module):
    """
    Anomaly Attention computing both Prior-Association (Gaussian)
    and Series-Association (Self-Attention).
    """

    def __init__(self, d_model: int = 64, n_heads: int = 4, seq_len: int = 60):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.seq_len = seq_len

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.sigma_proj = nn.Linear(d_model, n_heads)
        idx = torch.arange(seq_len)
        self.register_buffer("distances", (idx.unsqueeze(0) - idx.unsqueeze(1)).abs().float())

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L, _ = x.shape
        Q = self.q_proj(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, L, d_k)
        K = self.k_proj(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        # 1. Series Association (Self-Attention)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, H, L, L)
        series_attn = F.softmax(scores, dim=-1)

        # 2. Prior Association (Gaussian kernel parameterized by sigma)
        # sigma: (B, L, H) -> (B, H, L, 1)
        sigma = torch.sigmoid(self.sigma_proj(x)).transpose(1, 2).unsqueeze(-1) + 1e-4
        dist = self.distances[:L, :L].unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)
        # Gaussian prior: exp(-|i - j|^2 / (2 * sigma^2))
        prior_weights = torch.exp(-(dist**2) / (2 * (sigma**2) + 1e-6))
        prior_attn = prior_weights / (prior_weights.sum(dim=-1, keepdim=True) + 1e-8)

        # Compute attended values
        out = torch.matmul(series_attn, V)  # (B, H, L, d_k)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        out = self.out_proj(out)

        return out, series_attn, prior_attn


class AnomalyTransformerBlock(nn.Module):
    """Transformer block integrating Anomaly Attention and Feed Forward network."""

    def __init__(self, d_model: int = 64, n_heads: int = 4, seq_len: int = 60, d_ff: int = 128):
        super().__init__()
        self.attn = AnomalyAttention(d_model=d_model, n_heads=n_heads, seq_len=seq_len)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = x
        attn_out, series_attn, prior_attn = self.attn(x)
        x = self.norm1(residual + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x, series_attn, prior_attn


class AnomalyTransformer(nn.Module):
    """Complete 2-Layer Anomaly Transformer with Association Discrepancy."""

    def __init__(
        self,
        input_dim: int = 10,
        d_model: int = 64,
        n_heads: int = 4,
        seq_len: int = 60,
        num_layers: int = 2,
    ):
        super().__init__()
        self.input_dim, self.d_model, self.seq_len, self.num_layers = (
            input_dim,
            d_model,
            seq_len,
            num_layers,
        )
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_emb = PositionalEmbedding(d_model, max_len=max(500, seq_len))
        self.blocks = nn.ModuleList(
            [
                AnomalyTransformerBlock(d_model=d_model, n_heads=n_heads, seq_len=seq_len)
                for _ in range(num_layers)
            ]
        )
        self.output_proj = nn.Linear(d_model, input_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            reconstruction: (B, L, input_dim)
            series_list: list of series attentions
            prior_list: list of prior attentions
        """
        h = self.input_proj(x) + self.pos_emb(x)
        series_list = []
        prior_list = []
        for block in self.blocks:
            h, series_attn, prior_attn = block(h)
            series_list.append(series_attn)
            prior_list.append(prior_attn)
        recon = self.output_proj(h)
        return recon, series_list, prior_list


class AnomalyTransformerDetector:
    """Production wrapper for Anomaly Transformer training and anomaly scoring."""

    def __init__(
        self,
        input_dim: int = 10,
        d_model: int = 64,
        n_heads: int = 4,
        seq_len: int = 60,
        num_layers: int = 2,
        lr: float = 0.0001,
        k_val: float = 3.0,
        device: Optional[str] = None,
        checkpoint_path: str = "models/checkpoints/anomaly_transformer.pt",
    ):
        self.input_dim = input_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.lr = lr
        self.k_val = k_val
        self.checkpoint_path = checkpoint_path

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = AnomalyTransformer(
            input_dim=input_dim,
            d_model=d_model,
            n_heads=n_heads,
            seq_len=seq_len,
            num_layers=num_layers,
        ).to(self.device)

        self.threshold: float = 0.0
        self.is_fitted: bool = False

    def _association_discrepancy(self, series_list, prior_list) -> torch.Tensor:
        """Compute stable symmetric KL divergence between series and prior attention distributions."""
        total_kl = 0.0
        for series, prior in zip(series_list, prior_list):
            s_clamped = torch.clamp(series, min=1e-5, max=1.0)
            p_clamped = torch.clamp(prior, min=1e-5, max=1.0)
            kl1 = (s_clamped * (torch.log(s_clamped) - torch.log(p_clamped))).sum(dim=-1)
            kl2 = (p_clamped * (torch.log(p_clamped) - torch.log(s_clamped))).sum(dim=-1)
            total_kl += (kl1 + kl2) / 2.0  # (B, H, L)
        return total_kl.mean(dim=(1, 2))

    def fit(
        self,
        X_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        epochs: int = 15,
        batch_size: int = 32,
    ) -> "AnomalyTransformerDetector":
        """Train Anomaly Transformer with reconstruction and association discrepancy loss."""
        self.model.train()
        train_tensor = torch.tensor(X_train, dtype=torch.float32)
        dataset = TensorDataset(train_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        logger.info(
            f"Training Anomaly Transformer on {len(X_train)} samples for {epochs} epochs..."
        )
        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()
                recon, series_list, prior_list = self.model(batch_x)
                rec_loss = F.mse_loss(recon, batch_x)
                ass_dis = self._association_discrepancy(series_list, prior_list).mean()
                loss = rec_loss - 0.01 * ass_dis
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item() * len(batch_x)

            if epoch % 5 == 0 or epoch == 1:
                logger.info(
                    f"Anomaly Transformer Epoch [{epoch:02d}/{epochs:02d}] - Loss: {total_loss/len(X_train):.6f}"
                )

        self.save()
        self.calibrate_threshold(X_train)
        self.is_fitted = True
        return self

    def calibrate_threshold(self, X_normal: np.ndarray) -> float:
        """Calibrate statistical threshold on normal set."""
        scores = self.compute_anomaly_scores(X_normal)
        self.threshold = float(np.percentile(scores, 98.0))
        logger.info(f"Anomaly Transformer calibrated threshold: {self.threshold:.6f}")
        return self.threshold

    def compute_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Anomaly score = reconstruction error * association discrepancy."""
        self.model.eval()
        scores = []
        tensor_x = torch.tensor(X, dtype=torch.float32)
        loader = DataLoader(TensorDataset(tensor_x), batch_size=64, shuffle=False)
        with torch.no_grad():
            for (bx,) in loader:
                bx = bx.to(self.device)
                recon, series_list, prior_list = self.model(bx)
                rec_err = ((recon - bx) ** 2).mean(dim=(1, 2))  # (B,)
                ass_dis = self._association_discrepancy(series_list, prior_list)  # (B,)
                # Combined score
                batch_scores = rec_err * torch.sigmoid(ass_dis)
                scores.extend(batch_scores.cpu().numpy())
        return np.array(scores)

    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        thresh = threshold if threshold is not None else self.threshold
        scores = self.compute_anomaly_scores(X)
        return (scores > thresh).astype(int)

    def save(self, path: Optional[str] = None) -> str:
        target = path or self.checkpoint_path
        os.makedirs(os.path.dirname(target), exist_ok=True)
        ckpt = {
            "model_state_dict": self.model.state_dict(),
            "threshold": self.threshold,
            "input_dim": self.input_dim,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "seq_len": self.seq_len,
            "num_layers": self.num_layers,
        }
        torch.save(ckpt, target)
        return target

    def load(self, path: Optional[str] = None) -> "AnomalyTransformerDetector":
        target = path or self.checkpoint_path
        if not os.path.exists(target):
            raise FileNotFoundError(f"Checkpoint not found at {target}")
        checkpoint = torch.load(target, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.threshold = checkpoint.get("threshold", 0.0)
        self.is_fitted = True
        return self
