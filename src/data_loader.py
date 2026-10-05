"""
data_loader.py
--------------
Handles loading, preprocessing, and federated splitting of datasets:
  - FDI 118-Bus (synthetic)
  - UNSW-NB15
  - CIC IDS 2017
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# FDI Synthetic Dataset
# ---------------------------------------------------------------------------

def create_fdi_dataset(n_normal: int = 4000, n_attack: int = 2000, seed: int = 42):
    """
    Generate a realistic FDI 118-Bus synthetic dataset.
    Voltage magnitudes (118 buses) + phase angles (117 branches).
    """
    np.random.seed(seed)
    features, labels = [], []

    for _ in range(n_normal):
        v = np.random.normal(1.0, 0.02, 118)
        a = np.random.normal(0.0, 0.04, 117)
        features.append(np.concatenate([v, a]))
        labels.append(0)  # normal

    for _ in range(n_attack):
        v = np.random.normal(1.0, 0.02, 118)
        a = np.random.normal(0.0, 0.04, 117)
        attack_nodes = np.random.choice(118, 10, replace=False)
        v[attack_nodes] += 0.10 + np.random.normal(0, 0.01, size=10)
        features.append(np.concatenate([v, a]))
        labels.append(1)  # fdi_attack

    X = np.array(features)
    y = np.array(labels)
    return X, y


def load_unsw_nb15(data_dir: str = "UNSW NB15"):
    """Load and preprocess the UNSW-NB15 dataset."""
    train_path = os.path.join(data_dir, "UNSW_NB15_training-set.csv")
    test_path = os.path.join(data_dir, "UNSW_NB15_testing-set.csv")

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df = pd.concat([df_train, df_test], ignore_index=True)

    # Drop ID columns if present
    df.drop(columns=[c for c in ["id", "proto", "service", "state"] if c in df.columns],
            inplace=True, errors="ignore")

    label_col = "label" if "label" in df.columns else df.columns[-1]
    X = df.drop(columns=[label_col])
    y = df[label_col]

    # Encode categoricals
    for col in X.select_dtypes(include="object").columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    X = X.fillna(0).values.astype(np.float32)
    y = LabelEncoder().fit_transform(y)
    return X, y


def load_cic_ids2017(data_dir: str = "CIC IDS 2017"):
    """Load CIC-IDS-2017 parquet files and merge them."""
    files = [f for f in os.listdir(data_dir) if f.endswith(".parquet")]
    dfs = [pd.read_parquet(os.path.join(data_dir, f)) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    label_col = next((c for c in df.columns if "label" in c.lower()), df.columns[-1])
    X = df.drop(columns=[label_col])
    y = df[label_col]

    for col in X.select_dtypes(include="object").columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    X = X.fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    y = (LabelEncoder().fit_transform(y) > 0).astype(int)
    return X, y


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int = 42):
    """Scale features and split into train/test sets."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y, test_size=test_size, random_state=seed, stratify=y
    )
    return X_tr, X_te, y_tr, y_te, scaler


# ---------------------------------------------------------------------------
# Federated Splits
# ---------------------------------------------------------------------------

def make_federated_splits(X_tr: np.ndarray, y_tr: np.ndarray, n_clients: int = 5, seed: int = 42):
    """
    Partition training data among `n_clients` clients (IID split).
    Returns a list of (X_client, y_client) tuples.
    """
    np.random.seed(seed)
    indices = np.random.permutation(len(X_tr))
    chunks = np.array_split(indices, n_clients)
    return [(X_tr[idx], y_tr[idx]) for idx in chunks]
