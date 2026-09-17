"""
Isolation Forest Anomaly Detection Model for XAI for IOT Anomaly Detection.
Unsupervised tree-based anomaly detection baseline.
"""

import logging
import os
from typing import Optional

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)


class IsolationForestDetector:
    """Isolation Forest anomaly detection wrapper."""

    def __init__(
        self,
        n_estimators: int = 200,
        contamination: float = 0.04,
        random_state: int = 42,
        checkpoint_path: str = "models/checkpoints/isolation_forest.joblib",
    ):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.checkpoint_path = checkpoint_path
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.threshold: float = 0.0
        self.is_fitted: bool = False

    def _prepare_data(self, X: np.ndarray) -> np.ndarray:
        """Convert 3D sequence window (N, T, D) to 2D (N, D) using final timestep or mean."""
        if X.ndim == 3:
            return X[:, -1, :]  # take the latest timestep of the sequence
        return X

    def fit(self, X_train_normal: np.ndarray) -> "IsolationForestDetector":
        """Fit model on normal sensor readings only."""
        X_flat = self._prepare_data(X_train_normal)
        logger.info(f"Fitting IsolationForest on {X_flat.shape[0]} normal samples...")
        self.model.fit(X_flat)
        self.is_fitted = True

        # In IsolationForest, score_samples returns negative anomaly score (lower is more anomalous)
        # We invert so higher score indicates higher anomaly likelihood
        train_scores = self.compute_anomaly_scores(X_train_normal)
        self.threshold = float(np.percentile(train_scores, 100 * (1.0 - self.contamination)))
        logger.info(f"IsolationForest fitted. Calibrated threshold: {self.threshold:.5f}")
        return self

    def compute_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute continuous anomaly score (higher means more anomalous)."""
        if not self.is_fitted:
            raise RuntimeError("IsolationForest model must be fitted before scoring.")
        X_flat = self._prepare_data(X)
        # scikit-learn: score_samples is opposite of anomaly score
        scores = -self.model.score_samples(X_flat)
        return scores

    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """Predict binary anomaly flags (1=anomaly, 0=normal)."""
        thresh = threshold if threshold is not None else self.threshold
        scores = self.compute_anomaly_scores(X)
        return (scores > thresh).astype(int)

    def save(self, path: Optional[str] = None) -> str:
        """Save model and threshold to disk."""
        target_path = path or self.checkpoint_path
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        payload = {
            "model": self.model,
            "threshold": self.threshold,
            "n_estimators": self.n_estimators,
            "contamination": self.contamination,
            "random_state": self.random_state,
        }
        joblib.dump(payload, target_path)
        logger.info(f"Saved IsolationForest model to {target_path}")
        return target_path

    def load(self, path: Optional[str] = None) -> "IsolationForestDetector":
        """Load model and threshold from disk."""
        target_path = path or self.checkpoint_path
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Checkpoint not found at {target_path}")
        payload = joblib.load(target_path)
        self.model = payload["model"]
        self.threshold = payload["threshold"]
        self.n_estimators = payload["n_estimators"]
        self.contamination = payload["contamination"]
        self.random_state = payload["random_state"]
        self.is_fitted = True
        logger.info(f"Loaded IsolationForest from {target_path}")
        return self
