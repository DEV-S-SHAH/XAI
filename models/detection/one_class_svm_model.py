"""
One-Class Support Vector Machine (OC-SVM) Anomaly Detection Model.
Unsupervised boundary-based anomaly detection baseline.
"""

import logging
import os
from typing import Optional

import joblib
import numpy as np
from sklearn.svm import OneClassSVM

logger = logging.getLogger(__name__)


class OneClassSVMDetector:
    """One-Class SVM anomaly detection wrapper."""

    def __init__(
        self,
        kernel: str = "rbf",
        gamma: str = "scale",
        nu: float = 0.04,
        checkpoint_path: str = "models/checkpoints/one_class_svm.joblib",
    ):
        self.kernel = kernel
        self.gamma = gamma
        self.nu = nu
        self.checkpoint_path = checkpoint_path
        self.model = OneClassSVM(kernel=self.kernel, gamma=self.gamma, nu=self.nu)
        self.threshold: float = 0.0
        self.is_fitted: bool = False

    def _prepare_data(self, X: np.ndarray) -> np.ndarray:
        """Convert 3D sequence window (N, T, D) to 2D (N, D) using final timestep."""
        if X.ndim == 3:
            return X[:, -1, :]
        return X

    def fit(self, X_train_normal: np.ndarray) -> "OneClassSVMDetector":
        """Fit model on normal sensor readings only."""
        X_flat = self._prepare_data(X_train_normal)
        logger.info(f"Fitting OneClassSVM on {X_flat.shape[0]} normal samples...")
        self.model.fit(X_flat)
        self.is_fitted = True

        train_scores = self.compute_anomaly_scores(X_train_normal)
        self.threshold = float(np.percentile(train_scores, 100 * (1.0 - self.nu)))
        logger.info(f"OneClassSVM fitted. Calibrated threshold: {self.threshold:.5f}")
        return self

    def compute_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute continuous anomaly score (higher means more anomalous)."""
        if not self.is_fitted:
            raise RuntimeError("OneClassSVM model must be fitted before scoring.")
        X_flat = self._prepare_data(X)
        # decision_function gives positive for inliers, negative for outliers
        # We invert so higher score indicates anomaly
        scores = -self.model.decision_function(X_flat)
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
            "kernel": self.kernel,
            "gamma": self.gamma,
            "nu": self.nu,
        }
        joblib.dump(payload, target_path)
        logger.info(f"Saved OneClassSVM model to {target_path}")
        return target_path

    def load(self, path: Optional[str] = None) -> "OneClassSVMDetector":
        """Load model and threshold from disk."""
        target_path = path or self.checkpoint_path
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Checkpoint not found at {target_path}")
        payload = joblib.load(target_path)
        self.model = payload["model"]
        self.threshold = payload["threshold"]
        self.kernel = payload["kernel"]
        self.gamma = payload["gamma"]
        self.nu = payload["nu"]
        self.is_fitted = True
        logger.info(f"Loaded OneClassSVM from {target_path}")
        return self
