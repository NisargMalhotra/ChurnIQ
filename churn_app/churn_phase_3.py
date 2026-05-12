# ============================================================
# CUSTOMER CHURN PREDICTOR
# Phase 3 - Feature Engineering & Preprocessing
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

print("=" * 55)
print("  PHASE 3 — FEATURE ENGINEERING & PREPROCESSING")
print("=" * 55)

# ============================================================
# STEP 3.1 — Load & Repeat Essential Fixes from Phase 2
# ============================================================
print("\n── Step 3.1: Loading dataset ─────────────────────────")

df = pd.read_csv("customer_churn.csv")

# Fix TotalCharges (blank strings → 0)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(0, inplace=True)

# Drop customerID — not useful for modeling
df.drop(columns=["customerID"], inplace=True)

print(f"✅ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"   customerID dropped (not a feature)")

# ============================================================
# STEP 3.2 — Feature Engineering (Create New Features)
# ============================================================
print("\n── Step 3.2: Feature Engineering ────────────────────")

# 1. Charges per month (avoids multicollinearity between Monthly & Total)
#    Guard against divide-by-zero for tenure = 0
df["ChargesPerMonth"] = np.where(
    df["tenure"] > 0,
    df["TotalCharges"] / df["tenure"],
    df["MonthlyCharges"]
)

# 2. Tenure groups — buckets make patterns clearer
def tenure_group(tenure):
    if tenure <= 12:
        return "0-1 yr"
    elif tenure <= 24:
        return "1-2 yr"
    elif tenure <= 48:
        return "2-4 yr"
    else:
        return "4+ yr"

df["TenureGroup"] = df["tenure"].apply(tenure_group)

# 3. Number of services subscribed (more services = more engaged = less churn)
service_cols = [
    "PhoneService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
]
df["NumServices"] = df[service_cols].apply(
    lambda row: sum(val == "Yes" for val in row), axis=1
)

# 4. High value customer flag
monthly_median = df["MonthlyCharges"].median()
df["HighValueCustomer"] = (
    (df["MonthlyCharges"] > monthly_median) & (df["tenure"] > 24)
).astype(int)

print("✅ New features created:")
print("   ChargesPerMonth   — TotalCharges / tenure")
print("   TenureGroup       — Bucketed tenure (0-1yr, 1-2yr, 2-4yr, 4+yr)")
print("   NumServices       — Count of active services (0–7)")
print("   HighValueCustomer — High charges & long tenure (0/1)")

# Quick sanity check
print(f"\n   NumServices range : {df['NumServices'].min()} – {df['NumServices'].max()}")
print(f"   HighValueCustomer : {df['HighValueCustomer'].sum()} customers flagged")

# ============================================================
# STEP 3.3 — Encode Categorical Columns
# ============================================================
print("\n── Step 3.3: Encoding Categorical Columns ────────────")

# ── Binary columns (Yes/No or Male/Female) → Label Encode (0/1)
binary_cols = [
    "gender", "Partner", "Dependents", "PhoneService",
    "PaperlessBilling", "Churn",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies"
]

le = LabelEncoder()
for col in binary_cols:
    if col in df.columns:
        df[col] = le.fit_transform(df[col].astype(str))

print(f"✅ Label-encoded {len(binary_cols)} binary columns  (0 / 1)")

# ── Multi-class columns → One-Hot Encode
ohe_cols = ["MultipleLines", "InternetService", "Contract",
            "PaymentMethod", "TenureGroup"]

df = pd.get_dummies(df, columns=ohe_cols, drop_first=False)
print(f"✅ One-hot encoded  : {ohe_cols}")
print(f"   Dataset shape after encoding: {df.shape}")

# ── Drop TotalCharges (highly correlated with ChargesPerMonth & tenure)
df.drop(columns=["TotalCharges"], inplace=True)
print("✅ Dropped TotalCharges (multicollinearity — kept ChargesPerMonth)")

# ============================================================
# STEP 3.4 — Split Features & Target
# ============================================================
print("\n── Step 3.4: Train / Test Split ─────────────────────")

X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Train set : {X_train.shape[0]:,} rows")
print(f"   Test set  : {X_test.shape[0]:,} rows")
print(f"   Features  : {X_train.shape[1]}")
print(f"\n   Churn rate in train : {y_train.mean()*100:.1f}%")
print(f"   Churn rate in test  : {y_test.mean()*100:.1f}%")

# ============================================================
# STEP 3.5 — Scale Numeric Features
# ============================================================
print("\n── Step 3.5: Feature Scaling ─────────────────────────")

numeric_features = ["tenure", "MonthlyCharges", "ChargesPerMonth",
                    "NumServices", "HighValueCustomer", "SeniorCitizen"]

# Only scale columns that actually exist
numeric_features = [c for c in numeric_features if c in X_train.columns]

scaler = StandardScaler()
X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test[numeric_features]  = scaler.transform(X_test[numeric_features])

print(f"✅ StandardScaler applied to: {numeric_features}")
print("   (Fitted on train only — no data leakage)")

# ============================================================
# STEP 3.6 — Handle Class Imbalance with SMOTE
# ============================================================
print("\n── Step 3.6: Handling Class Imbalance with SMOTE ────")

print(f"   Before SMOTE → Churn=0: {(y_train==0).sum():,}  |  Churn=1: {(y_train==1).sum():,}")

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print(f"   After  SMOTE → Churn=0: {(y_train_sm==0).sum():,}  |  Churn=1: {(y_train_sm==1).sum():,}")
print("✅ Classes balanced — models won't be biased toward 'No Churn'")

# ============================================================
# PLOT — Feature Distributions Before/After Engineering
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# NumServices distribution
axes[0].hist(df["NumServices"], bins=8, color="#4C9BE8", edgecolor="white")
axes[0].set_title("NumServices Distribution", fontweight="bold")
axes[0].set_xlabel("Number of Services"); axes[0].set_ylabel("Count")

# TenureGroup vs Churn (reload raw to plot)
raw = pd.read_csv("customer_churn.csv")
raw["TotalCharges"] = pd.to_numeric(raw["TotalCharges"], errors="coerce").fillna(0)
raw["TenureGroup"] = raw["tenure"].apply(tenure_group)
raw["Churn_num"] = (raw["Churn"] == "Yes").astype(int)
order = ["0-1 yr", "1-2 yr", "2-4 yr", "4+ yr"]
churn_by_group = raw.groupby("TenureGroup")["Churn_num"].mean() * 100
churn_by_group = churn_by_group.reindex(order)
axes[1].bar(churn_by_group.index, churn_by_group.values,
            color="#E8734C", edgecolor="white")
axes[1].set_title("Churn % by Tenure Group", fontweight="bold")
axes[1].set_xlabel("Tenure Group"); axes[1].set_ylabel("Churn %")
for i, v in enumerate(churn_by_group.values):
    axes[1].text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")

# Class balance after SMOTE
smote_counts = pd.Series(y_train_sm).value_counts()
axes[2].bar(["No Churn", "Churn"], smote_counts.values,
            color=["#4C9BE8", "#E8734C"], edgecolor="white")
axes[2].set_title("Class Balance After SMOTE", fontweight="bold")
axes[2].set_ylabel("Count")
for i, v in enumerate(smote_counts.values):
    axes[2].text(i, v + 20, str(v), ha="center", fontweight="bold")

plt.suptitle("Phase 3 — Feature Engineering Insights", fontsize=13,
             fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("plot9_phase3_features.png", bbox_inches="tight")
plt.show()
print("\n📊 Plot 9 saved: Phase 3 Feature Engineering Insights")

# ============================================================
# STEP 3.7 — Save Processed Data for Phase 4
# ============================================================
print("\n── Step 3.7: Saving Processed Data ──────────────────")

joblib.dump(X_train_sm,       "X_train.pkl")
joblib.dump(y_train_sm,       "y_train.pkl")
joblib.dump(X_test,           "X_test.pkl")
joblib.dump(y_test,           "y_test.pkl")
joblib.dump(scaler,           "scaler.pkl")
joblib.dump(X_train.columns.tolist(), "feature_names.pkl")

print("✅ Saved:")
print("   X_train.pkl, y_train.pkl  — SMOTE-balanced training data")
print("   X_test.pkl,  y_test.pkl   — Original test data (no SMOTE)")
print("   scaler.pkl                — Fitted StandardScaler")
print("   feature_names.pkl         — Column names for SHAP in Phase 5")

# ============================================================
# PHASE 3 SUMMARY
# ============================================================
print("\n" + "=" * 55)
print("  ✅ PHASE 3 COMPLETE — SUMMARY")
print("=" * 55)
print(f"""
Features after engineering : {X_train_sm.shape[1]}
Training samples (SMOTE)   : {X_train_sm.shape[0]:,}
Test samples               : {X_test.shape[0]:,}

Steps completed:
  ✅ Fixed TotalCharges (blank → 0)
  ✅ Dropped customerID
  ✅ Created 4 new features (ChargesPerMonth, TenureGroup,
     NumServices, HighValueCustomer)
  ✅ Label-encoded binary columns
  ✅ One-hot encoded multi-class columns
  ✅ Dropped TotalCharges (multicollinearity)
  ✅ Train/test split (80/20, stratified)
  ✅ StandardScaler on numeric features (no leakage)
  ✅ SMOTE applied to fix class imbalance

Next → Phase 4: Model Building & Comparison
""")
