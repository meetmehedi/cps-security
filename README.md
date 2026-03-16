# CPS Security: Federated Learning & XAI

This repository contains research code and results for an advanced Security Framework for Cyber-Physical Systems (CPS), focusing on **False Data Injection (FDI)** detection and **Explainable AI (XAI)**.

## Project Overview
The project implements a **Federated Learning (FL)** framework to robustly detect attacks across heterogeneous datasets (Power Systems and Network Traffic) while maintaining data privacy. High-performance classical ML baselines (SVM, XGBoost, Random Forest, Decision Tree) are also provided for comparative analysis.

### Key Features
- **Heterogeneous Federated Learning**: Supports unified training across diverse CPS datasets.
- **Explainable AI (XAI)**: Integrated **SHAP** and **LIME** for model transparency.
- **High-Precision Detection**: Achieves **99.87% Accuracy** and **0.999 AUC** on FDI detection.
- **Multi-Dataset Support**: Integrated with world-standard datasets:
  - TON_IoT
  - UNSW-NB15
  - CIC IDS 2017 & 2018
  - FDI 118-Bus (Refined Synthetic)

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
1.  Ensure you have the required datasets in the root directory (see `.gitignore`).
2.  Install dependencies: `pip install torch xgboost shap lime scikit-learn pandas numpy matplotlib seaborn`
3.  Open and run the main notebook: `XAI & FL ..Asif .ipynb`

## License
MIT License
