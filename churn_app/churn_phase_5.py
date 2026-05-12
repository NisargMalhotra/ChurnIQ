# ============================================================
# CUSTOMER CHURN PREDICTOR
# Phase 5 - SHAP Interpretability
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import warnings

warnings.filterwarnings("ignore")

print("=" * 55)
print("  PHASE 5 — SHAP INTERPRETABILITY")
print("=" * 55)

# ============================================================
# STEP 5.1 — Load Best Model & Test Data
# ============================================================
print("\n── Step 5.1: Loading Model & Data ───────────────────")

best_model      = joblib.load("best_model.pkl")
best_model_name = joblib.load("best_model_name.pkl")
X_test          = joblib.load("X_test.pkl")
y_test          = joblib.load("y_test.pkl")
feature_names   = joblib.load("feature_names.pkl")

# Ensure X_test is a DataFrame with correct column names
X_test = pd.DataFrame(X_test, columns=feature_names)

print(f"✅ Model loaded   : {best_model_name}")
print(f"   Test samples  : {X_test.shape[0]:,}")
print(f"   Features      : {X_test.shape[1]}")

# ============================================================
# STEP 5.2 — Build SHAP Explainer
# ============================================================
print("\n── Step 5.2: Building SHAP Explainer ────────────────")

if best_model_name in ["XGBoost", "LightGBM"]:
    explainer   = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test)

    # LightGBM returns a list [class0, class1] — pick class 1 (Churn)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

else:
    # Logistic Regression
    explainer   = shap.LinearExplainer(best_model, X_test)
    shap_values = explainer.shap_values(X_test)

print(f"✅ SHAP explainer built  ({type(explainer).__name__})")

# ── Fix: force shap_values to a proper float64 NumPy array ──
# Needed for compatibility with NumPy 1.24+ and SHAP on Python 3.13
shap_values = np.array(shap_values, dtype=np.float64)

print(f"   SHAP values shape : {shap_values.shape}")

# ============================================================
# PLOT 14 — SHAP Summary Plot (Bar) — Global Feature Importance
# ============================================================
print("\n── Plot 14: SHAP Summary Bar Plot ───────────────────")

plt.figure()
shap.summary_plot(
    shap_values, X_test,
    plot_type="bar",
    feature_names=feature_names,
    max_display=15,
    show=False
)
plt.title("Plot 14 — Global Feature Importance (SHAP)", fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig("plot14_shap_bar.png", bbox_inches="tight")
plt.show()
print("📊 Plot 14 saved: SHAP Global Feature Importance (Bar)")
print("   → Shows which features influence churn predictions the MOST overall")

# ============================================================
# PLOT 15 — SHAP Summary Dot Plot — Direction of Impact
# ============================================================
print("\n── Plot 15: SHAP Summary Dot Plot ───────────────────")

plt.figure()
shap.summary_plot(
    shap_values, X_test,
    feature_names=feature_names,
    max_display=15,
    show=False
)
plt.title("Plot 15 — Feature Impact Direction (SHAP Dot Plot)", fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig("plot15_shap_dot.png", bbox_inches="tight")
plt.show()
print("📊 Plot 15 saved: SHAP Dot Plot")
print("   → Red  = high feature value  → pushes prediction toward Churn")
print("   → Blue = low feature value   → pushes prediction away from Churn")

# ============================================================
# PLOT 16 — SHAP Waterfall Plot — Why ONE customer will churn
# ============================================================
print("\n── Plot 16: SHAP Waterfall Plot (Single Prediction) ─")

# Find a customer predicted as high churn risk
mean_shap      = np.abs(shap_values).mean(axis=1)
high_risk_idx  = int(np.argmax(mean_shap))

# Get expected value safely
if isinstance(explainer.expected_value, (list, np.ndarray)):
    expected_val = float(explainer.expected_value[1])
else:
    expected_val = float(explainer.expected_value)

# Build Explanation object for waterfall
explanation = shap.Explanation(
    values        = shap_values[high_risk_idx],
    base_values   = expected_val,
    data          = X_test.iloc[high_risk_idx].values,
    feature_names = feature_names
)

plt.figure()
shap.waterfall_plot(explanation, max_display=12, show=False)
plt.title(f"Plot 16 — Why Customer #{high_risk_idx} is Predicted to Churn",
          fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig("plot16_shap_waterfall.png", bbox_inches="tight")
plt.show()

actual_label = "Churn ✅" if y_test.iloc[high_risk_idx] == 1 else "No Churn"
print(f"📊 Plot 16 saved: SHAP Waterfall for Customer #{high_risk_idx}")
print(f"   Actual label for this customer : {actual_label}")
print("   → Each bar shows how much a feature pushed the prediction")
print("      up (toward churn) or down (away from churn)")

# ============================================================
# PLOT 17 — SHAP Dependence Plot — Tenure vs Churn Risk
# ============================================================
print("\n── Plot 17: SHAP Dependence Plot (Tenure) ───────────")

# Find tenure index safely
if "tenure" in feature_names:
    tenure_idx = feature_names.index("tenure")

    plt.figure(figsize=(9, 5))
    shap.dependence_plot(
        tenure_idx,
        shap_values,
        X_test,
        feature_names=feature_names,
        show=False
    )
    plt.title("Plot 17 — SHAP Dependence: Tenure vs Churn Risk",
              fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig("plot17_shap_dependence_tenure.png", bbox_inches="tight")
    plt.show()
    print("📊 Plot 17 saved: SHAP Dependence Plot — Tenure")
    print("   → Shows how tenure value affects churn risk (SHAP value)")
    print("   → Color shows interaction with another feature (auto-selected)")
else:
    print("⚠️  'tenure' column not found — skipping dependence plot")

# ============================================================
# PLOT 18 — SHAP Dependence Plot — MonthlyCharges vs Churn Risk
# ============================================================
print("\n── Plot 18: SHAP Dependence Plot (MonthlyCharges) ───")

if "MonthlyCharges" in feature_names:
    mc_idx = feature_names.index("MonthlyCharges")

    plt.figure(figsize=(9, 5))
    shap.dependence_plot(
        mc_idx,
        shap_values,
        X_test,
        feature_names=feature_names,
        show=False
    )
    plt.title("Plot 18 — SHAP Dependence: Monthly Charges vs Churn Risk",
              fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig("plot18_shap_dependence_charges.png", bbox_inches="tight")
    plt.show()
    print("📊 Plot 18 saved: SHAP Dependence Plot — Monthly Charges")
    print("   → Higher monthly charges generally push churn risk up")
else:
    print("⚠️  'MonthlyCharges' column not found — skipping dependence plot")

# ============================================================
# STEP 5.3 — Print Business Insights from SHAP
# ============================================================
print("\n── Step 5.3: Key Business Insights ──────────────────")

# Top 5 most impactful features globally
mean_abs_shap = np.abs(shap_values).mean(axis=0)
top5_idx      = np.argsort(mean_abs_shap)[::-1][:5]
top5_features = [(feature_names[i], round(mean_abs_shap[i], 4)) for i in top5_idx]

print("\n  Top 5 features driving churn predictions:")
for rank, (feat, score) in enumerate(top5_features, 1):
    print(f"    {rank}. {feat:35s}  SHAP importance = {score}")

print("""
  Business Recommendations (based on SHAP findings):

  📌 Target new customers early
     → Low tenure is a top churn driver. Engage customers
       in months 1–12 with onboarding offers & check-ins.

  📌 Push annual/two-year contracts
     → Month-to-month customers churn the most. Offer
       discounts to lock customers into longer contracts.

  📌 Review high monthly charge customers
     → High charges + low tenure = highest risk profile.
       Consider loyalty pricing or service bundles.

  📌 Promote OnlineSecurity & TechSupport add-ons
     → Customers without these churn significantly more.
       Bundle them into starter plans.

  📌 Flag electronic check users
     → This payment method correlates strongly with churn.
       Incentivize auto-pay (credit card / bank transfer).
""")

# ============================================================
# PHASE 5 SUMMARY
# ============================================================
print("=" * 55)
print("  ✅ PHASE 5 COMPLETE — SUMMARY")
print("=" * 55)
print(f"""
Model explained   : {best_model_name}
SHAP explainer    : {"TreeExplainer" if best_model_name in ["XGBoost", "LightGBM"] else "LinearExplainer"}
Test samples used : {X_test.shape[0]:,}

Plots saved (5 files):
  plot14_shap_bar.png              — Global feature importance
  plot15_shap_dot.png              — Feature impact direction
  plot16_shap_waterfall.png        — Single customer explanation
  plot17_shap_dependence_tenure.png     — Tenure vs churn risk
  plot18_shap_dependence_charges.png    — Charges vs churn risk

🎯 Interview tip:
   "I used SHAP TreeExplainer on my best model to explain
   both global patterns (which features matter most) and
   individual predictions (why a specific customer churns).
   This bridges the gap between ML and business decisions."

Next → Phase 6: README + GitHub Documentation
""")
