# ============================================================
# CUSTOMER CHURN PREDICTOR
# Phase 4 - Model Building & Comparison
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

print("=" * 55)
print("  PHASE 4 — MODEL BUILDING & COMPARISON")
print("=" * 55)

# ============================================================
# STEP 4.1 — Load Processed Data from Phase 3
# ============================================================
print("\n── Step 4.1: Loading Processed Data ─────────────────")

X_train = joblib.load("X_train.pkl")
y_train = joblib.load("y_train.pkl")
X_test  = joblib.load("X_test.pkl")
y_test  = joblib.load("y_test.pkl")

print(f"✅ Train : {X_train.shape[0]:,} rows × {X_train.shape[1]} features")
print(f"   Test  : {X_test.shape[0]:,} rows  × {X_test.shape[1]} features")

# ============================================================
# STEP 4.2 — Define Models
# ============================================================
print("\n── Step 4.2: Defining Models ─────────────────────────")

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        verbose=-1
    )
}

print("✅ 3 models defined:")
print("   1. Logistic Regression  (baseline — simple & interpretable)")
print("   2. XGBoost              (gradient boosting — strong on tabular)")
print("   3. LightGBM             (faster boosting — often best on tabular)")

# ============================================================
# STEP 4.3 — Train & Evaluate All Models
# ============================================================
print("\n── Step 4.3: Training & Evaluating ──────────────────")

results    = {}   # metrics table
roc_data   = {}   # for ROC curve plot
cm_data    = {}   # for confusion matrices

for name, model in models.items():
    print(f"\n  ▶ Training {name} ...")

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    results[name] = {
        "Accuracy":  round(acc  * 100, 2),
        "Precision": round(prec * 100, 2),
        "Recall":    round(rec  * 100, 2),
        "F1 Score":  round(f1   * 100, 2),
        "ROC-AUC":   round(auc  * 100, 2),
    }
    roc_data[name] = (fpr, tpr, auc)
    cm_data[name]  = confusion_matrix(y_test, y_pred)

    print(f"    Accuracy  : {acc*100:.2f}%")
    print(f"    Precision : {prec*100:.2f}%")
    print(f"    Recall    : {rec*100:.2f}%")
    print(f"    F1 Score  : {f1*100:.2f}%")
    print(f"    ROC-AUC   : {auc*100:.2f}%")

# ============================================================
# STEP 4.4 — Comparison Table
# ============================================================
print("\n── Step 4.4: Model Comparison Table ─────────────────")

results_df = pd.DataFrame(results).T
results_df = results_df.sort_values("ROC-AUC", ascending=False)
print("\n" + results_df.to_string())

best_model_name = results_df.index[0]
best_model      = models[best_model_name]
print(f"\n🏆 Best model → {best_model_name}  (ROC-AUC: {results_df.loc[best_model_name,'ROC-AUC']}%)")

# ============================================================
# PLOT 10 — Model Comparison Bar Chart
# ============================================================
metrics   = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
model_names = results_df.index.tolist()
x    = np.arange(len(metrics))
w    = 0.25
colors = ["#4C9BE8", "#E8734C", "#4CAF7D"]

fig, ax = plt.subplots(figsize=(13, 5))
for i, (mname, color) in enumerate(zip(model_names, colors)):
    vals = [results_df.loc[mname, m] for m in metrics]
    bars = ax.bar(x + i * w, vals, width=w, label=mname,
                  color=color, edgecolor="white", linewidth=1)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=7.5)

ax.set_xticks(x + w)
ax.set_xticklabels(metrics)
ax.set_ylabel("Score (%)")
ax.set_ylim(50, 105)
ax.set_title("Plot 10 — Model Comparison (All Metrics)", fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig("plot10_model_comparison.png", bbox_inches="tight")
plt.show()
print("\n📊 Plot 10 saved: Model Comparison Bar Chart")

# ============================================================
# PLOT 11 — ROC Curves (All 3 Models)
# ============================================================
plt.figure(figsize=(8, 6))
colors_roc = ["#4C9BE8", "#E8734C", "#4CAF7D"]

for (name, (fpr, tpr, auc)), color in zip(roc_data.items(), colors_roc):
    plt.plot(fpr, tpr, label=f"{name}  (AUC = {auc:.3f})",
             color=color, linewidth=2)

plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("Plot 11 — ROC Curves", fontweight="bold")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("plot11_roc_curves.png", bbox_inches="tight")
plt.show()
print("📊 Plot 11 saved: ROC Curves")

# ============================================================
# PLOT 12 — Confusion Matrices (All 3 Models)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, (name, cm) in zip(axes, cm_data.items()):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"],
                linewidths=0.5, cbar=False)
    ax.set_title(f"{name}", fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")

    tn, fp, fn, tp = cm.ravel()
    ax.set_xlabel(
        f"Predicted\nTP={tp}  FP={fp}  FN={fn}  TN={tn}", fontsize=8
    )

plt.suptitle("Plot 12 — Confusion Matrices", fontsize=13,
             fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("plot12_confusion_matrices.png", bbox_inches="tight")
plt.show()
print("📊 Plot 12 saved: Confusion Matrices")

# ============================================================
# PLOT 13 — Feature Importance (Best Model)
# ============================================================
print(f"\n── Feature Importance ({best_model_name}) ────────────────────")

if best_model_name in ["XGBoost", "LightGBM"]:
    feature_names = joblib.load("feature_names.pkl")
    importances   = best_model.feature_importances_
    feat_df = pd.DataFrame({
        "Feature":    feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False).head(15)

    plt.figure(figsize=(10, 6))
    bars = plt.barh(feat_df["Feature"][::-1],
                    feat_df["Importance"][::-1],
                    color="#4C9BE8", edgecolor="white")
    plt.xlabel("Feature Importance Score")
    plt.title(f"Plot 13 — Top 15 Features ({best_model_name})", fontweight="bold")
    plt.tight_layout()
    plt.savefig("plot13_feature_importance.png", bbox_inches="tight")
    plt.show()
    print("📊 Plot 13 saved: Feature Importance")

else:
    # Logistic Regression — use coefficients
    feature_names = joblib.load("feature_names.pkl")
    coefs = pd.DataFrame({
        "Feature":     feature_names,
        "Coefficient": best_model.coef_[0]
    })
    coefs["abs"] = coefs["Coefficient"].abs()
    coefs = coefs.sort_values("abs", ascending=False).head(15)

    colors_lr = ["#E8734C" if c > 0 else "#4C9BE8"
                 for c in coefs["Coefficient"][::-1]]
    plt.figure(figsize=(10, 6))
    plt.barh(coefs["Feature"][::-1], coefs["Coefficient"][::-1],
             color=colors_lr, edgecolor="white")
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Coefficient Value  (red = increases churn risk)")
    plt.title("Plot 13 — Top 15 Features (Logistic Regression)", fontweight="bold")
    plt.tight_layout()
    plt.savefig("plot13_feature_importance.png", bbox_inches="tight")
    plt.show()
    print("📊 Plot 13 saved: Logistic Regression Coefficients")

# ============================================================
# STEP 4.5 — Save Best Model
# ============================================================
print("\n── Step 4.5: Saving Best Model ───────────────────────")

joblib.dump(best_model, "best_model.pkl")
joblib.dump(best_model_name, "best_model_name.pkl")

print(f"✅ Saved: best_model.pkl        ({best_model_name})")
print(f"   Saved: best_model_name.pkl")

# ============================================================
# PHASE 4 SUMMARY
# ============================================================
print("\n" + "=" * 55)
print("  ✅ PHASE 4 COMPLETE — SUMMARY")
print("=" * 55)
print(f"""
Models trained    : Logistic Regression, XGBoost, LightGBM
Best model        : {best_model_name}
Best ROC-AUC      : {results_df.loc[best_model_name, 'ROC-AUC']}%

Full results:
{results_df.to_string()}

Plots saved (4 files):
  plot10_model_comparison.png
  plot11_roc_curves.png
  plot12_confusion_matrices.png
  plot13_feature_importance.png

💡 Why ROC-AUC matters most for churn:
   Churn detection is about ranking customers by risk.
   A high AUC means the model correctly separates churners
   from non-churners, even with class imbalance.

💡 Why Recall matters too:
   Missing a churner (False Negative) costs more than a
   false alarm — the business loses that customer forever.

Next → Phase 5: SHAP Interpretability
""")
