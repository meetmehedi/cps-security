# CPS Security: Federated Learning & XAI

This repository contains research code and results for an advanced Security Framework for Cyber-Physical Systems (CPS), focusing on **False Data Injection (FDI)** detection and **Explainable AI (XAI)**.

## Project Overview

The project implements a **Federated Learning (FL)** framework using the **FedAvg** algorithm to robustly detect attacks across heterogeneous datasets (Power Systems and Network Traffic) while maintaining data privacy. High-performance classical ML baselines (SVM, XGBoost, Random Forest, Decision Tree) are provided for comparative analysis, with full SHAP and LIME explainability.

### Key Features

- **Federated Averaging (FedAvg)**: Full multi-client FL implementation with a PyTorch MLP backbone and configurable rounds, clients, and local epochs.
- **Explainable AI (XAI)**: Integrated **SHAP** (global) and **LIME** (local) for model transparency.
- **Classical ML Baselines**: SVM, XGBoost, Random Forest, Decision Tree with ROC-AUC analysis.
- **High-Precision Detection**: Achieves **99.87% Accuracy** and **0.999 AUC** on FDI detection.
- **Multi-Dataset Support**: Works with world-standard datasets:
  - TON_IoT
  - UNSW-NB15
  - CIC IDS 2017 & 2018
  - FDI 118-Bus (Refined Synthetic)

## Project Structure

```
cps-security/
├── src/
│   ├── __init__.py
│   ├── data_loader.py    # Dataset loading, preprocessing & federated splits
│   ├── models.py         # PyTorch MLP + classical ML model definitions
│   ├── federated.py      # FedAvg: Client, Server, and training orchestrator
│   └── xai_utils.py      # SHAP, LIME, and convergence plot utilities
├── results/              # Generated plots and metrics
├── main.py               # CLI pipeline entry point
├── requirements.txt
└── XAI & FL ..Asif .ipynb  # Interactive notebook
```

## Results

Detailed performance metrics, confusion matrices, and ROC-AUC curves are available in the `results/` directory.

### Comprehensive Performance Metrics

| Dataset | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **TON_IoT** | 99.52% | 99.50% | 99.52% | 99.51% |
| **UNSW-NB15**| 99.33% | 99.32% | 99.33% | 99.30% |
| **CIC IDS 17**| 98.14% | 98.10% | 98.14% | 97.72% |
| **CIC IDS 18**| 99.61% | 99.60% | 99.61% | 99.54% |
| **FDI 118-Bus**| 99.87% | 99.87% | 99.87% | 99.87% |

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Full Pipeline (CLI)

```bash
# Full pipeline on synthetic FDI dataset (fastest, no external data needed)
python main.py

# Run on UNSW-NB15 (requires dataset files in ./UNSW NB15/)
python main.py --dataset unsw

# Run on CIC-IDS-2017 (requires parquet files in ./CIC IDS 2017/)
python main.py --dataset cic

# Customize FL hyperparameters
python main.py --rounds 30 --clients 10 --local-epochs 5 --lr 0.001
```

### 3. Interactive Notebook

Open and run the Jupyter notebook for an interactive walkthrough:

```bash
jupyter notebook "XAI & FL ..Asif .ipynb"
```

> **Note**: Large dataset files (`.csv`, `.parquet`) are excluded via `.gitignore`. Download them separately and place them in the appropriate directories before running.

## License

MIT License
