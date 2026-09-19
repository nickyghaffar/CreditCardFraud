import os
import sys
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Force UTF-8 encoding
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

app = FastAPI(title="Credit Card Fraud AI Engine", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load scalers and models
SCALER_PATH = "models/scalers.joblib"
DATA_PATH = "data/processed_data.joblib"
MODELS_PATH = "models/all_trained_models.joblib"
BENCHMARK_PATH = "models/benchmark_results.csv"
SAMPLE_DATA_PATH = "data/sample_transactions.csv"
RAW_DATA_PATH = "data/creditcard.csv"

print("Loading preprocessing scalers and trained models...")
if os.path.exists(SCALER_PATH):
    scalers_bundle = joblib.load(SCALER_PATH)
    scaler_amount = scalers_bundle["scaler_amount"]
    scaler_time = scalers_bundle["scaler_time"]
    expected_features = scalers_bundle["expected_features"]
elif os.path.exists(DATA_PATH):
    processed_bundle = joblib.load(DATA_PATH)
    scaler_amount = processed_bundle["scaler_amount"]
    scaler_time = processed_bundle["scaler_time"]
    expected_features = processed_bundle["X_train"].columns.tolist()
else:
    raise FileNotFoundError("Neither scalers.joblib nor processed_data.joblib found.")

trained_models = joblib.load(MODELS_PATH)
# Add fallback for Random Forest if needed
if "5. Random Forest" not in trained_models and os.path.exists("models/random_forest_model.joblib"):
    trained_models["5. Random Forest"] = joblib.load("models/random_forest_model.joblib")

benchmark_df = pd.read_csv(BENCHMARK_PATH) if os.path.exists(BENCHMARK_PATH) else pd.DataFrame()

if os.path.exists(RAW_DATA_PATH):
    raw_df = pd.read_csv(RAW_DATA_PATH)
elif os.path.exists(SAMPLE_DATA_PATH):
    raw_df = pd.read_csv(SAMPLE_DATA_PATH)
else:
    raw_df = pd.DataFrame()

print(f"Loaded {len(trained_models)} models and scalers successfully.")

class TransactionPayload(BaseModel):
    model_name: str = "9. XGBoost"
    Time: float = 85000.0
    Amount: float = 125.50
    V1: float = -0.5
    V2: float = 0.2
    V3: float = 0.1
    V4: float = 0.0
    V5: float = 0.0
    V6: float = 0.0
    V7: float = 0.0
    V8: float = 0.0
    V9: float = 0.0
    V10: float = -0.1
    V11: float = 0.1
    V12: float = -0.2
    V13: float = 0.0
    V14: float = -0.3
    V15: float = 0.0
    V16: float = 0.0
    V17: float = -0.2
    V18: float = 0.0
    V19: float = 0.0
    V20: float = 0.0
    V21: float = 0.0
    V22: float = 0.0
    V23: float = 0.0
    V24: float = 0.0
    V25: float = 0.0
    V26: float = 0.0
    V27: float = 0.0
    V28: float = 0.0

@app.get("/api/models")
def get_available_models():
    """Return list of models with their benchmark performance"""
    models_info = []
    for _, row in benchmark_df.iterrows():
        models_info.append({
            "name": row["Model"],
            "pr_auc": float(row["PR-AUC"]),
            "f1": float(row["F1-Score"]),
            "precision": float(row["Precision"]),
            "recall": float(row["Recall"]),
            "train_time": float(row["Train Time (s)"])
        })
    return {"models": models_info}

@app.get("/api/benchmark")
def get_benchmark_table():
    return benchmark_df.to_dict(orient="records")

@app.get("/api/sample/{sample_type}")
def get_sample_transaction(sample_type: str):
    """Provide realistic sample data for quick testing in UI"""
    if sample_type == "normal":
        row = raw_df[raw_df["Class"] == 0].iloc[1050].to_dict()
    elif sample_type == "fraud_blatant":
        # High impact fraud
        row = raw_df[raw_df["Class"] == 1].iloc[12].to_dict()
    elif sample_type == "fraud_micro":
        # Card testing $1 fraud
        row = raw_df[(raw_df["Class"] == 1) & (raw_df["Amount"] <= 2.0)].iloc[5].to_dict()
    else:
        row = raw_df.sample(1).iloc[0].to_dict()
        
    actual_class = int(row.pop("Class", 0))
    return {"features": row, "actual_class": actual_class}

@app.get("/api/stream")
def get_stream_batch(count: int = 8):
    """Simulate a stream of incoming real-time transactions with mix of normal and fraud"""
    normal_samples = raw_df[raw_df["Class"] == 0].sample(max(1, count - 2))
    fraud_samples = raw_df[raw_df["Class"] == 1].sample(min(2, count))
    combined = pd.concat([normal_samples, fraud_samples]).sample(frac=1.0).reset_index(drop=True)
    
    stream_list = []
    for i, r in combined.iterrows():
        d = r.to_dict()
        act = int(d.pop("Class", 0))
        stream_list.append({
            "tx_id": f"TXN-{np.random.randint(100000, 999999)}",
            "amount": float(d["Amount"]),
            "time": float(d["Time"]),
            "actual": "FRAUD" if act == 1 else "Normal",
            "features": d
        })
    return {"transactions": stream_list}

@app.post("/api/predict")
def predict_fraud(payload: TransactionPayload):
    model_key = payload.model_name
    
    # Handle model selection
    clf = None
    if model_key in trained_models:
        clf = trained_models[model_key]
    else:
        # Match by prefix/name
        for k in trained_models:
            if model_key.lower() in k.lower():
                clf = trained_models[k]
                break
                
    if clf is None:
        clf = trained_models.get("9. XGBoost", list(trained_models.values())[0])

    data_dict = payload.model_dump()
    data_dict.pop("model_name", None)

    input_df = pd.DataFrame([data_dict])

    # Scale Time and Amount
    input_df["scaled_amount"] = scaler_amount.transform(input_df[["Amount"]])
    input_df["scaled_time"] = scaler_time.transform(input_df[["Time"]])
    input_df = input_df.drop(columns=["Time", "Amount"])

    # Ensure feature alignment
    features_aligned = input_df[expected_features]

    # Predict
    if hasattr(clf, "predict_proba"):
        prob = float(clf.predict_proba(features_aligned)[:, 1][0])
    else:
        # Super ensemble soft-voting fallback
        prob = float((
            trained_models["5. Random Forest"].predict_proba(features_aligned)[:, 1][0] +
            trained_models["9. XGBoost"].predict_proba(features_aligned)[:, 1][0] +
            trained_models["10. LightGBM"].predict_proba(features_aligned)[:, 1][0]
        ) / 3.0)

    # Risk evaluation
    if prob >= 0.50:
        risk_level = "CRITICAL"
        risk_label_fa = "خطر کلاهبرداری بالا (مسدودسازی)"
        action = "BLOCK TRANSACTION & ALERT CARDHOLDER"
        badge_color = "#ff3860"
    elif prob >= 0.20:
        risk_level = "MEDIUM"
        risk_label_fa = "مشکوک (نیاز به تایید دوعاملی SMS)"
        action = "REQUIRE 2FA OTP VERIFICATION"
        badge_color = "#ffdd57"
    else:
        risk_level = "SAFE"
        risk_label_fa = "تراکنش امن و معتبر"
        action = "APPROVE TRANSACTION"
        badge_color = "#00d1b2"

    return {
        "model_used": model_key,
        "fraud_probability": round(prob, 4),
        "fraud_percentage": round(prob * 100, 2),
        "risk_level": risk_level,
        "risk_label_fa": risk_label_fa,
        "action": action,
        "badge_color": badge_color,
        "is_fraud": prob >= 0.50
    }

# Mount plots folder so all charts are accessible by browser
if os.path.exists("plots"):
    app.mount("/plots", StaticFiles(directory="plots"), name="plots")

# Mount static frontend
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_index():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
