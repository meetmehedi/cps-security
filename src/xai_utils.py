"""
xai_utils.py
------------
Reusable SHAP and LIME explanation utilities for CPS security models.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import lime
import lime.lime_tabular


# ---------------------------------------------------------------------------
# SHAP Utilities
# ---------------------------------------------------------------------------

def compute_shap_values(model, X_sample: np.ndarray, model_type: str = "tree"):
    """
    Compute SHAP values for the given model and samples.

    Args:
        model:       Trained model (XGBoost, RF, or torch MLP).
        X_sample:    Array of shape (n_samples, n_features).
        model_type:  'tree' for tree-based models, 'kernel' for MLP/SVM.

    Returns:
        explainer, shap_values
    """
    if model_type == "tree":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    else:
        # KernelExplainer works for any black-box model
        background = shap.sample(X_sample, min(100, len(X_sample)), random_state=42)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_sample[:50], nsamples=100)

    return explainer, shap_values


def plot_shap_summary(shap_values, X_sample: np.ndarray, feature_names: list,
                      save_path: str = None, title: str = "SHAP Feature Importance"):
    """
    Generate and optionally save a SHAP beeswarm summary plot.
    """
    # Handle multi-class shap_values (take class-1 slice)
    vals = shap_values[1] if isinstance(shap_values, list) else shap_values

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        vals, X_sample,
        feature_names=feature_names,
        show=False,
        plot_size=None,
    )
    plt.title(title, fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[XAI] SHAP summary saved → {save_path}")

    plt.close()


def plot_shap_bar(shap_values, feature_names: list,
                  save_path: str = None, top_n: int = 20):
    """Plot mean absolute SHAP values as a horizontal bar chart."""
    vals = shap_values[1] if isinstance(shap_values, list) else shap_values
    mean_abs = np.abs(vals).mean(axis=0)

    indices = np.argsort(mean_abs)[-top_n:]
    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    ax.barh(
        [feature_names[i] for i in indices],
        mean_abs[indices],
        color="#4C72B0",
    )
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title("Top Feature Importances (SHAP)", fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[XAI] SHAP bar chart saved → {save_path}")

    plt.close()


# ---------------------------------------------------------------------------
# LIME Utilities
# ---------------------------------------------------------------------------

def explain_instance_lime(
    model,
    X_train: np.ndarray,
    X_instance: np.ndarray,
    class_names: list,
    feature_names: list,
    num_features: int = 10,
    save_path: str = None,
):
    """
    Generate a LIME local explanation for a single instance.

    Args:
        model:         Trained classifier with `predict_proba`.
        X_train:       Training data (for LIME background distribution).
        X_instance:    Single sample of shape (n_features,).
        class_names:   List of class label strings.
        feature_names: List of feature name strings.
        num_features:  Number of top features to show.
        save_path:     If given, save the explanation figure to this path.

    Returns:
        exp: LimeTabularExplanation object.
    """
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train,
        mode="classification",
        class_names=class_names,
        feature_names=feature_names,
        random_state=42,
    )

    exp = explainer.explain_instance(
        X_instance,
        model.predict_proba,
        num_features=num_features,
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig = exp.as_pyplot_figure()
        fig.suptitle("LIME Local Explanation", fontweight="bold")
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[XAI] LIME explanation saved → {save_path}")

    return exp


# ---------------------------------------------------------------------------
# Federated Learning Convergence Plot
# ---------------------------------------------------------------------------

def plot_fl_convergence(history: list, save_path: str = None):
    """
    Plot FL accuracy and loss curves across communication rounds.

    Args:
        history:   List of per-round metric dicts (from run_federated_training).
        save_path: If given, save the figure to this path.
    """
    rounds = [h["round"] for h in history]
    accs = [h["accuracy"] * 100 for h in history]
    losses = [h["avg_client_loss"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(rounds, accs, marker="o", color="#2ecc71", linewidth=2, markersize=4)
    ax1.set_title("FL Global Accuracy per Round", fontweight="bold")
    ax1.set_xlabel("Communication Round")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim([max(0, min(accs) - 5), 101])
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot(rounds, losses, marker="s", color="#e74c3c", linewidth=2, markersize=4)
    ax2.set_title("Avg Client Training Loss per Round", fontweight="bold")
    ax2.set_xlabel("Communication Round")
    ax2.set_ylabel("BCE Loss")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[XAI] Convergence plot saved → {save_path}")

    plt.close()
