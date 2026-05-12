# ============================================================
# CUSTOMER CHURN PREDICTOR — FastAPI Backend
# ============================================================

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import pandas as pd
import numpy as np
import joblib
import shap
import io
import os
import warnings
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

app = FastAPI(title="Customer Churn Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# ── Globals (loaded model persists across requests) ──────────
MODEL_PATH = "best_model.pkl"
SCALER_PATH = "scaler.pkl"
FEATURE_PATH = "feature_names.pkl"

# ============================================================
# HELPERS
# ============================================================

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def preprocess(df: pd.DataFrame):
    """Full preprocessing pipeline matching Phase 3."""
    df = df.copy()

    # Drop ID column if present
    for col in ["customerID", "customer_id", "id"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Fix TotalCharges
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    # Drop target if present
    target_col = None
    for col in ["Churn", "churn"]:
        if col in df.columns:
            target_col = col
            break

    y = None
    if target_col:
        col_vals = df[target_col].astype(str).str.strip().str.lower()
        y = col_vals.map({"yes": 1, "no": 0, "1": 1, "0": 0}).fillna(0).astype(int)
        df.drop(columns=[target_col], inplace=True)
        
    # Feature engineering
    if "tenure" in df.columns and "TotalCharges" in df.columns:
        df["ChargesPerMonth"] = np.where(df["tenure"] > 0, df["TotalCharges"] / df["tenure"], df.get("MonthlyCharges", 0))
    elif "MonthlyCharges" in df.columns:
        df["ChargesPerMonth"] = df["MonthlyCharges"]

    def tenure_group(t):
        if t <= 12:   return "0-1 yr"
        elif t <= 24: return "1-2 yr"
        elif t <= 48: return "2-4 yr"
        else:         return "4+ yr"

    if "tenure" in df.columns:
        df["TenureGroup"] = df["tenure"].apply(tenure_group)

    service_cols = ["PhoneService","OnlineSecurity","OnlineBackup",
                    "DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]
    present_services = [c for c in service_cols if c in df.columns]
    df["NumServices"] = df[present_services].apply(lambda r: sum(v == "Yes" for v in r), axis=1)

    if "MonthlyCharges" in df.columns and "tenure" in df.columns:
        median_charge = df["MonthlyCharges"].median()
        df["HighValueCustomer"] = ((df["MonthlyCharges"] > median_charge) & (df["tenure"] > 24)).astype(int)

    # Binary encode
    binary_cols = ["gender","Partner","Dependents","PhoneService","PaperlessBilling",
                   "OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport",
                   "StreamingTV","StreamingMovies"]
    le = LabelEncoder()
    for col in binary_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    # One-hot encode
    ohe_cols = ["MultipleLines","InternetService","Contract","PaymentMethod","TenureGroup"]
    ohe_cols = [c for c in ohe_cols if c in df.columns]
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=False)

    # Drop TotalCharges
    if "TotalCharges" in df.columns:
        df.drop(columns=["TotalCharges"], inplace=True)

    return df, y


def align_features(df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """Add missing columns as 0, drop extra columns, reorder."""
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    return df[feature_names]


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def root():
    return {"status": "Churn Predictor API running"}


@app.get("/model-status")
def model_status():
    exists = os.path.exists(MODEL_PATH)
    name = None
    if exists:
        try:
            name = joblib.load("best_model_name.pkl")
        except:
            name = "Unknown"
    return {"model_loaded": exists, "model_name": name}


# ── PREDICT with existing model ──────────────────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=400, detail="No trained model found. Please train first.")

    contents = await file.read()
    df_raw = pd.read_csv(io.BytesIO(contents))

    # Save customer name/ID for display
    id_col = next((c for c in ["customerID","customer_id","name","Name","id"] if c in df_raw.columns), None)
    customer_ids = df_raw[id_col].tolist() if id_col else [f"Customer #{i+1}" for i in range(len(df_raw))]

    df_processed, y_true = preprocess(df_raw.copy())

    model        = joblib.load(MODEL_PATH)
    scaler       = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURE_PATH)

    df_aligned = align_features(df_processed, feature_names)

    numeric_features = ["tenure","MonthlyCharges","ChargesPerMonth","NumServices","HighValueCustomer","SeniorCitizen"]
    numeric_features = [c for c in numeric_features if c in df_aligned.columns]
    df_aligned[numeric_features] = scaler.transform(df_aligned[numeric_features])

    proba = model.predict_proba(df_aligned)[:, 1]
    preds = (proba >= 0.5).astype(int)

    # ── Build customer table ─────────────────────────────────
    raw_for_display = df_raw.copy()
    results = []
    for i, (cid, prob, pred) in enumerate(zip(customer_ids, proba, preds)):
        row = {
            "id": cid,
            "churn_probability": round(float(prob) * 100, 1),
            "prediction": "Will Churn" if pred == 1 else "Will Stay",
            "risk_level": "High" if prob >= 0.7 else ("Medium" if prob >= 0.4 else "Low"),
            "monthly_charges": float(raw_for_display.get("MonthlyCharges", pd.Series([0])).iloc[i]) if "MonthlyCharges" in raw_for_display.columns else None,
            "tenure": int(raw_for_display["tenure"].iloc[i]) if "tenure" in raw_for_display.columns else None,
            "contract": raw_for_display["Contract"].iloc[i] if "Contract" in raw_for_display.columns else None,
        }
        results.append(row)

    results.sort(key=lambda x: x["churn_probability"], reverse=True)

    # ── Summary stats ────────────────────────────────────────
    total        = len(preds)
    churn_count  = int(preds.sum())
    stay_count   = total - churn_count
    high_risk    = sum(1 for r in results if r["risk_level"] == "High")
    medium_risk  = sum(1 for r in results if r["risk_level"] == "Medium")
    low_risk     = sum(1 for r in results if r["risk_level"] == "Low")
    avg_prob     = round(float(proba.mean()) * 100, 1)
    revenue_at_risk = round(sum(
        r["monthly_charges"] for r in results
        if r["prediction"] == "Will Churn" and r["monthly_charges"] is not None
    ), 2)

    # ── Charts ───────────────────────────────────────────────
    charts = {}

    # Chart 1: Risk distribution donut
    fig, ax = plt.subplots(figsize=(5, 5))
    sizes  = [high_risk, medium_risk, low_risk]
    labels = [f"High Risk\n{high_risk}", f"Medium Risk\n{medium_risk}", f"Low Risk\n{low_risk}"]
    colors = ["#EF4444", "#F59E0B", "#22C55E"]
    wedges, texts = ax.pie(sizes, labels=labels, colors=colors,
                           startangle=90, wedgeprops={"edgecolor":"white","linewidth":2},
                           textprops={"fontsize": 11})
    ax.set_title("Risk Distribution", fontweight="bold", fontsize=13)
    charts["risk_donut"] = fig_to_base64(fig)

    # Chart 2: Churn vs Stay bar
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Will Stay", "Will Churn"], [stay_count, churn_count],
           color=["#22C55E", "#EF4444"], edgecolor="white", linewidth=1.5, width=0.5)
    for i, v in enumerate([stay_count, churn_count]):
        ax.text(i, v + 0.5, str(v), ha="center", fontweight="bold", fontsize=12)
    ax.set_title("Predicted Churn vs Stay", fontweight="bold", fontsize=13)
    ax.set_ylabel("Number of Customers")
    ax.set_facecolor("#F9FAFB"); fig.patch.set_facecolor("#F9FAFB")
    charts["churn_bar"] = fig_to_base64(fig)

    # Chart 3: Probability histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(proba * 100, bins=20, color="#6366F1", edgecolor="white", linewidth=0.8)
    ax.axvline(50, color="#EF4444", linestyle="--", linewidth=1.5, label="50% threshold")
    ax.set_xlabel("Churn Probability (%)"); ax.set_ylabel("Number of Customers")
    ax.set_title("Churn Probability Distribution", fontweight="bold", fontsize=13)
    ax.legend(); ax.set_facecolor("#F9FAFB"); fig.patch.set_facecolor("#F9FAFB")
    charts["prob_hist"] = fig_to_base64(fig)

    # Chart 4: SHAP summary (top 10 features)
    try:
        model_name = joblib.load("best_model_name.pkl")
        if model_name in ["XGBoost", "LightGBM"]:
            explainer   = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(df_aligned)
            if isinstance(shap_values, list): shap_values = shap_values[1]
        else:
            explainer   = shap.LinearExplainer(model, df_aligned)
            shap_values = explainer.shap_values(df_aligned)
        shap_values = np.array(shap_values, dtype=np.float64)

        fig, ax = plt.subplots(figsize=(8, 5))
        mean_shap = np.abs(shap_values).mean(axis=0)
        top10_idx = np.argsort(mean_shap)[::-1][:10]
        top10_names = [feature_names[i] for i in top10_idx]
        top10_vals  = [mean_shap[i] for i in top10_idx]
        ax.barh(top10_names[::-1], top10_vals[::-1], color="#6366F1", edgecolor="white")
        ax.set_xlabel("Mean |SHAP Value|")
        ax.set_title("Top 10 Features Driving Churn", fontweight="bold", fontsize=13)
        ax.set_facecolor("#F9FAFB"); fig.patch.set_facecolor("#F9FAFB")
        charts["shap_bar"] = fig_to_base64(fig)
    except Exception as e:
        charts["shap_bar"] = None

    return {
        "summary": {
            "total_customers": total,
            "churn_count": churn_count,
            "stay_count": stay_count,
            "churn_rate": round(churn_count / total * 100, 1),
            "avg_churn_probability": avg_prob,
            "high_risk": high_risk,
            "medium_risk": medium_risk,
            "low_risk": low_risk,
            "revenue_at_risk": revenue_at_risk,
        },
        "customers": results,
        "charts": charts,
    }


# ── RETRAIN + PREDICT ────────────────────────────────────────
@app.post("/retrain")
async def retrain(file: UploadFile = File(...)):
    contents = await file.read()
    df_raw   = pd.read_csv(io.BytesIO(contents))

    # Check target column exists
    target_col = next((c for c in ["Churn","churn"] if c in df_raw.columns), None)
    if not target_col:
        raise HTTPException(status_code=400, detail="Dataset must have a 'Churn' column to retrain.")

    df_processed, y = preprocess(df_raw.copy())
    if y is None:
        raise HTTPException(status_code=400, detail="Could not extract target column.")

    feature_names = df_processed.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        df_processed, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric_features = [c for c in ["tenure","MonthlyCharges","ChargesPerMonth",
                        "NumServices","HighValueCustomer","SeniorCitizen"] if c in X_train.columns]
    scaler = StandardScaler()
    X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_test[numeric_features]  = scaler.transform(X_test[numeric_features])

    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5,
                                  use_label_encoder=False, eval_metric="logloss", random_state=42),
        "LightGBM": LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=5,
                                    random_state=42, verbose=-1),
    }

    best_name, best_model, best_auc = None, None, 0
    all_results = {}

    for name, model in models.items():
        model.fit(X_train_sm, y_train_sm)
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        all_results[name] = {
            "accuracy":  round(accuracy_score(y_test, y_pred) * 100, 2),
            "precision": round(precision_score(y_test, y_pred) * 100, 2),
            "recall":    round(recall_score(y_test, y_pred) * 100, 2),
            "f1":        round(f1_score(y_test, y_pred) * 100, 2),
            "roc_auc":   round(auc * 100, 2),
        }
        if auc > best_auc:
            best_auc, best_name, best_model = auc, name, model

    joblib.dump(best_model,    MODEL_PATH)
    joblib.dump(best_name,     "best_model_name.pkl")
    joblib.dump(scaler,        SCALER_PATH)
    joblib.dump(feature_names, FEATURE_PATH)

    return {
        "message": f"Retrained successfully. Best model: {best_name}",
        "best_model": best_name,
        "best_roc_auc": round(best_auc * 100, 2),
        "all_results": all_results,
        "training_rows": len(X_train_sm),
        "test_rows": len(X_test),
        "features": len(feature_names),
    }
