"""
models.py
---------
Neural network and classical ML model definitions for CPS security.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


# ---------------------------------------------------------------------------
# PyTorch MLP (used by Federated Learning clients)
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """
    Multi-Layer Perceptron for binary classification.
    Architecture: input -> 256 -> 128 -> 64 -> 1 (sigmoid)
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)

    def get_weights(self) -> list:
        """Return a deep copy of model parameter tensors."""
        return [p.data.clone() for p in self.parameters()]

    def set_weights(self, weights: list):
        """Load parameter tensors into the model."""
        for p, w in zip(self.parameters(), weights):
            p.data.copy_(w)


# ---------------------------------------------------------------------------
# Classical ML Baselines
# ---------------------------------------------------------------------------

def get_classical_models() -> dict:
    """Return a dictionary of configured classical ML models."""
    return {
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        ),
        "SVM (RBF)": SVC(kernel="rbf", C=1.0, probability=True, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            verbosity=0,
            random_state=42,
            eval_metric="logloss",
        ),
    }
