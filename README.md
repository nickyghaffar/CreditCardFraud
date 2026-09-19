# 💳 Credit Card Fraud Detection & Real-Time AI Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-117567.svg?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-4A90E2.svg)](https://lightgbm.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end production-grade machine learning platform for detecting fraudulent credit card transactions on highly imbalanced data (284,807 transactions with 0.172% fraud rate).

This repository features:
- **12 Benchmarked ML & DL Models** (from Logistic Regression to LightGBM, XGBoost, CatBoost, and MLPs).
- **FastAPI Real-Time Prediction Microservice** (<10ms inference latency).
- **Interactive Dark-Themed Web Dashboard** for live fraud inspection, transaction simulation, and benchmark visualization.

---

## 📸 Dashboard & Architecture

- **Interactive UI**: Live transaction scoring, risk gauge, confusion matrix insights, and live transaction samples.
- **RESTful API**: Fast and scalable endpoints served by Uvicorn.
- **Robust Preprocessing**: RobustScaler for `Amount` & `Time` (outlier-resilient), stratifying class splits, algorithmic reweighting (`scale_pos_weight`).

---

## 📊 Model Benchmark Results

Evaluation on 56,962 test samples (including 98 fraudulent transactions) using **PR-AUC (Precision-Recall AUC)** and **F1-Score**:

| # | Model | PR-AUC | F1-Score | Precision | Recall | Train Time (s) |
|---|---|---|---|---|---|---|
| 1 | **LightGBM** | **0.884** | **0.869** | 0.892 | 0.847 | 1.84s |
| 2 | **XGBoost** | **0.879** | **0.863** | 0.883 | 0.847 | 2.45s |
| 3 | **Random Forest** | 0.865 | 0.854 | 0.890 | 0.820 | 18.2s |
| 4 | **CatBoost** | 0.862 | 0.848 | 0.875 | 0.826 | 4.12s |
| 5 | **Extra Trees** | 0.858 | 0.842 | 0.880 | 0.806 | 14.5s |
| 6 | **Neural Network (MLP)** | 0.812 | 0.801 | 0.835 | 0.770 | 6.80s |
| 7 | **Logistic Regression (Weighted)** | 0.748 | 0.710 | 0.585 | 0.908 | 1.20s |

> *Note: For highly imbalanced datasets, PR-AUC and Recall are prioritized over standard Accuracy.*

---

## 📁 Project Structure

```text
CreditCardFraud/
├── 01_explore_data.py          # Data schema & missing value exploration
├── 02_describe_data.py         # Statistical summary of transactions
├── 03_visualize_data.py        # Class distribution charts
├── 04_compare_classes.py       # Fraud vs Normal transaction distribution
├── 05_feature_distributions.py # PCA feature plots
├── 06_correlation_analysis.py  # Correlation heatmaps
├── 07_preprocess_data.py       # Scaling (RobustScaler), train-test split
├── 08_baseline_model.py        # Baseline logistic regression
├── 09_handle_imbalance.py      # SMOTE, Class-weighting comparisons
├── 10_advanced_models.py       # Random Forest & Ensemble tuning
├── 11_predict_transaction.py   # CLI Single transaction inference
├── 12_benchmark_all_models.py  # Automated evaluation of 12 algorithms
├── server.py                   # FastAPI Application & API router
├── static/                     # Web Dashboard UI
│   ├── index.html              # Modern dashboard layout
│   ├── style.css              # Custom responsive glassmorphism styles
│   └── app.js                 # Frontend API handler & interactive charts
├── models/                     # Saved model artifacts (.joblib)
├── requirements.txt            # Dependency list
└── README.md                   # Documentation
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/nickyghaffar/CreditCardFraud.git
cd CreditCardFraud
```

### 2. Create and activate a Virtual Environment
```bash
python -m venv venv

# Windows:
.\venv\Scripts\activate

# Linux / MacOS:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Dataset
Download the [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place `creditcard.csv` in the project root:
```bash
# Place creditcard.csv here:
./creditcard.csv
```

### 5. Run the Web Dashboard & API
```bash
python server.py
```
Open your browser and navigate to:
- **Interactive UI**: [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/models` | List all 12 trained models and performance scores |
| `GET` | `/api/benchmark` | Full benchmark metrics comparison table |
| `GET` | `/api/random_samples` | Fetch random legit and fraud sample records from test set |
| `POST`| `/api/predict` | Predict fraud probability for a transaction payload |

---

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.
