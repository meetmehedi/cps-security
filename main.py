#!/usr/bin/env python3
"""
main.py
-------
CLI entry point to run the full CPS Security pipeline:
  1. Dataset preparation
  2. Classical ML baselines (with ROC-AUC)
  3. Federated Learning (FedAvg)
  4. Explainable AI (SHAP + LIME)

Usage:
    python main.py                    # Run full pipeline on FDI dataset
    python main.py --dataset unsw     # Run on UNSW-NB15
    python main.py --dataset cic      # Run on CIC-IDS-2017
    python main.py --rounds 10        # Custom FL rounds
    python main.py --clients 5        # Custom number of FL clients
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_curve, auc, ConfusionMatrixDisplay, confusion_matrix,
)

from src.data_loader import (
    create_fdi_dataset, load_unsw_nb15, load_cic_ids2017,
    preprocess, make_federated_splits,
)
from src.models import MLP, get_classical_models
from src.federated import run_federated_training
from src.xai_utils import (
    compute_shap_values, plot_shap_summary, plot_shap_bar,
    explain_instance_lime, plot_fl_convergence,
)

RESULTS_DIR = "results"


def parse_args():
    parser = argparse.ArgumentParser(description="CPS Security: FL + XAI Pipeline")
    parser.add_argument(
        "--dataset", choices=["fdi", "unsw", "cic"], default="fdi",
        help="Dataset to use (default: fdi)",
    )
    parser.add_argument("--rounds", type=int, default=20, help="FL communication rounds")
    parser.add_argument("--clients", type=int, default=5, help="Number of FL clients")
    parser.add_argument("--local-epochs", type=int, default=3, help="Local epochs per client")
    parser.add_argument("--lr", type=float, default=1e-3, help="Client learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Step 1: Data Loading
# ---------------------------------------------------------------------------
def load_data(dataset: str, seed: int):
    print(f"\n{'='*60}")
    print(f"  STEP 1: LOADING DATASET → {dataset.upper()}")
    print(f"{'='*60}")

    if dataset == "fdi":
        X, y = create_fdi_dataset(seed=seed)
        feature_names = [f"V_{i}" for i in range(118)] + [f"A_{i}" for i in range(117)]
    elif dataset == "unsw":
        X, y = load_unsw_nb15()
        feature_names = [f"f_{i}" for i in range(X.shape[1])]
    else:
        X, y = load_cic_ids2017()
        feature_names = [f"f_{i}" for i in range(X.shape[1])]

    print(f"  Samples: {X.shape[0]:,}  |  Features: {X.shape[1]}  |  Attack rate: {y.mean():.1%}")
    X_tr, X_te, y_tr, y_te, _ = preprocess(X, y, seed=seed)
    return X_tr, X_te, y_tr, y_te, feature_names


# ---------------------------------------------------------------------------
# Step 2: Classical ML Baselines
# ---------------------------------------------------------------------------
def run_classical(X_tr, X_te, y_tr, y_te):
    print(f"\n{'='*60}")
    print("  STEP 2: CLASSICAL ML BASELINES")
    print(f"{'='*60}")

    models = get_classical_models()
    results = {}

    plt.figure(figsize=(10, 8))
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        probs = model.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, probs)
        roc_auc = auc(fpr, tpr)

        acc = accuracy_score(y_te, preds)
        f1 = f1_score(y_te, preds, zero_division=0)
        prec = precision_score(y_te, preds, zero_division=0)
        rec = recall_score(y_te, preds, zero_division=0)

        results[name] = {"model": model, "acc": acc, "f1": f1, "prec": prec, "rec": rec, "auc": roc_auc}
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={roc_auc:.3f})")
        print(f"  {name:<20} Acc={acc:.4f}  F1={f1:.4f}  AUC={roc_auc:.3f}")

    plt.plot([0, 1], [0, 1], "k--", lw=1.5)
    plt.title("ROC-AUC: Classical ML Baselines", fontsize=14, fontweight="bold")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    save = os.path.join(RESULTS_DIR, "roc_auc_classical.png")
    plt.savefig(save, dpi=150)
    plt.close()
    print(f"\n  [✓] ROC-AUC plot saved → {save}")
    return results


# ---------------------------------------------------------------------------
# Step 3: Federated Learning
# ---------------------------------------------------------------------------
def run_fl(X_tr, X_te, y_tr, y_te, n_clients, n_rounds, local_epochs, lr):
    print(f"\n{'='*60}")
    print(f"  STEP 3: FEDERATED LEARNING  ({n_clients} clients, {n_rounds} rounds)")
    print(f"{'='*60}")

    splits = make_federated_splits(X_tr, y_tr, n_clients=n_clients)
    server, history = run_federated_training(
        client_splits=splits,
        X_te=X_te,
        y_te=y_te,
        n_rounds=n_rounds,
        local_epochs=local_epochs,
        lr=lr,
        verbose=True,
    )

    final = history[-1]
    print(f"\n  [✓] Final FL | Acc={final['accuracy']:.4f}  F1={final['f1']:.4f}")

    conv_path = os.path.join(RESULTS_DIR, "fl_convergence.png")
    plot_fl_convergence(history, save_path=conv_path)
    return server, history


# ---------------------------------------------------------------------------
# Step 4: XAI
# ---------------------------------------------------------------------------
def run_xai(classical_results, X_tr, X_te, y_te, feature_names):
    print(f"\n{'='*60}")
    print("  STEP 4: EXPLAINABLE AI (SHAP + LIME)")
    print(f"{'='*60}")

    xgb_model = classical_results["XGBoost"]["model"]
    svm_model = classical_results["SVM (RBF)"]["model"]

    # SHAP on XGBoost
    print("  Computing SHAP values (XGBoost)...")
    _, shap_vals = compute_shap_values(xgb_model, X_te[:200], model_type="tree")
    plot_shap_summary(
        shap_vals, X_te[:200], feature_names,
        save_path=os.path.join(RESULTS_DIR, "shap_summary.png"),
        title="SHAP Summary — XGBoost",
    )
    plot_shap_bar(
        shap_vals, feature_names,
        save_path=os.path.join(RESULTS_DIR, "shap_bar.png"),
    )

    # LIME on SVM — explain a correctly classified attack
    attack_idx = np.where(y_te == 1)[0][0]
    print("  Computing LIME explanation (SVM)...")
    explain_instance_lime(
        model=svm_model,
        X_train=X_tr,
        X_instance=X_te[attack_idx],
        class_names=["normal", "attack"],
        feature_names=feature_names,
        num_features=15,
        save_path=os.path.join(RESULTS_DIR, "lime_explanation.png"),
    )

    print("  [✓] XAI plots saved to results/")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.random.seed(args.seed)

    print("\n" + "="*60)
    print("  CPS Security: Federated Learning + XAI Pipeline")
    print("="*60)

    X_tr, X_te, y_tr, y_te, feature_names = load_data(args.dataset, args.seed)
    classical_results = run_classical(X_tr, X_te, y_tr, y_te)
    server, history = run_fl(
        X_tr, X_te, y_tr, y_te,
        n_clients=args.clients,
        n_rounds=args.rounds,
        local_epochs=args.local_epochs,
        lr=args.lr,
    )
    run_xai(classical_results, X_tr, X_te, y_te, feature_names)

    print("\n" + "="*60)
    print("  [✓] Pipeline complete. Results saved to ./results/")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
