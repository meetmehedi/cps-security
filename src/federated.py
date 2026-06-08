"""
federated.py
------------
Federated Averaging (FedAvg) implementation for CPS security.

References:
  McMahan et al., "Communication-Efficient Learning of Deep Networks
  from Decentralized Data", AISTATS 2017.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple

from src.models import MLP


# ---------------------------------------------------------------------------
# Federated Client
# ---------------------------------------------------------------------------

class FederatedClient:
    """Simulates a single FL client with a local dataset partition."""

    def __init__(
        self,
        client_id: int,
        X: np.ndarray,
        y: np.ndarray,
        input_dim: int,
        device: torch.device,
        lr: float = 1e-3,
        local_epochs: int = 3,
        batch_size: int = 64,
    ):
        self.client_id = client_id
        self.device = device
        self.local_epochs = local_epochs

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        dataset = TensorDataset(X_t, y_t)
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model = MLP(input_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.criterion = nn.BCELoss()

    def set_global_weights(self, weights: list):
        """Sync local model with global server weights."""
        self.model.set_weights(weights)

    def local_train(self) -> Tuple[list, float]:
        """Run local training for `local_epochs` and return updated weights + loss."""
        self.model.train()
        total_loss = 0.0
        for _ in range(self.local_epochs):
            for X_batch, y_batch in self.loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                preds = self.model(X_batch)
                loss = self.criterion(preds, y_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

        avg_loss = total_loss / (len(self.loader) * self.local_epochs)
        return self.model.get_weights(), avg_loss

    def evaluate(self, X_te: np.ndarray, y_te: np.ndarray) -> dict:
        """Evaluate the local model on test data."""
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_te, dtype=torch.float32).to(self.device)
            preds = self.model(X_t).cpu().numpy()
        pred_labels = (preds >= 0.5).astype(int)

        acc = np.mean(pred_labels == y_te)
        tp = np.sum((pred_labels == 1) & (y_te == 1))
        fp = np.sum((pred_labels == 1) & (y_te == 0))
        fn = np.sum((pred_labels == 0) & (y_te == 1))
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return {
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "probs": preds,
        }


# ---------------------------------------------------------------------------
# Federated Server
# ---------------------------------------------------------------------------

class FederatedServer:
    """
    Central server that orchestrates FL rounds using FedAvg aggregation.
    """

    def __init__(self, input_dim: int, device: torch.device):
        self.device = device
        self.global_model = MLP(input_dim).to(device)

    def get_global_weights(self) -> list:
        return self.global_model.get_weights()

    def aggregate(self, client_weights: List[list], client_sizes: List[int]):
        """
        FedAvg: weighted average of client model weights.
        Weight of each client is proportional to its dataset size.
        """
        total = sum(client_sizes)
        new_weights = []

        for layer_idx in range(len(client_weights[0])):
            layer_avg = sum(
                w[layer_idx] * (n / total)
                for w, n in zip(client_weights, client_sizes)
            )
            new_weights.append(layer_avg)

        self.global_model.set_weights(new_weights)

    def evaluate(self, X_te: np.ndarray, y_te: np.ndarray) -> dict:
        """Evaluate the global model on the held-out test set."""
        self.global_model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_te, dtype=torch.float32).to(self.device)
            probs = self.global_model(X_t).cpu().numpy()

        pred_labels = (probs >= 0.5).astype(int)
        acc = np.mean(pred_labels == y_te)
        tp = np.sum((pred_labels == 1) & (y_te == 1))
        fp = np.sum((pred_labels == 1) & (y_te == 0))
        fn = np.sum((pred_labels == 0) & (y_te == 1))
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return {
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "probs": probs,
        }


# ---------------------------------------------------------------------------
# Training Orchestrator
# ---------------------------------------------------------------------------

def run_federated_training(
    client_splits: list,
    X_te: np.ndarray,
    y_te: np.ndarray,
    n_rounds: int = 20,
    local_epochs: int = 3,
    lr: float = 1e-3,
    device: torch.device = None,
    verbose: bool = True,
) -> Tuple[FederatedServer, list]:
    """
    Run the full FedAvg training loop.

    Args:
        client_splits: List of (X_client, y_client) tuples.
        X_te, y_te:    Global test set for evaluation.
        n_rounds:      Number of FL communication rounds.
        local_epochs:  Local training epochs per round.
        lr:            Client learning rate.
        device:        Torch device.
        verbose:       Print per-round metrics.

    Returns:
        server:  Trained FederatedServer.
        history: List of per-round metric dicts.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = client_splits[0][0].shape[1]
    server = FederatedServer(input_dim, device)

    clients = [
        FederatedClient(
            client_id=i,
            X=X,
            y=y,
            input_dim=input_dim,
            device=device,
            lr=lr,
            local_epochs=local_epochs,
        )
        for i, (X, y) in enumerate(client_splits)
    ]

    history = []

    for rnd in range(1, n_rounds + 1):
        global_weights = server.get_global_weights()
        client_weights, client_sizes, round_losses = [], [], []

        for client in clients:
            client.set_global_weights(copy.deepcopy(global_weights))
            weights, loss = client.local_train()
            client_weights.append(weights)
            client_sizes.append(len(client.loader.dataset))
            round_losses.append(loss)

        server.aggregate(client_weights, client_sizes)
        metrics = server.evaluate(X_te, y_te)
        metrics["round"] = rnd
        metrics["avg_client_loss"] = float(np.mean(round_losses))
        history.append(metrics)

        if verbose:
            print(
                f"Round {rnd:3d}/{n_rounds} | "
                f"Acc: {metrics['accuracy']:.4f} | "
                f"F1: {metrics['f1']:.4f} | "
                f"Loss: {metrics['avg_client_loss']:.4f}"
            )

    return server, history
