# ============================================================
# CUSTOMER CHURN PREDICTOR
# Phase 1 - Setup & Dataset | Phase 2 - EDA
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# ── Plot style ───────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["figure.figsize"] = (10, 5)

# ============================================================
# PHASE 1 — SETUP & DATASET
# ============================================================

print("=" * 55)
print("  PHASE 1 — SETUP & DATASET")
print("=" * 55)

# ── 1.1  Load data ───────────────────────────────────────────
df = pd.read_csv("customer_churn.csv")
print(f"\n✅ Dataset loaded  →  {df.shape[0]:,} rows, {df.shape[1]} columns\n")

# ── 1.2  First look ──────────────────────────────────────────
print("── First 5 rows ──────────────────────────────────────")
print(df.head())

print("\n── Column names ──────────────────────────────────────")
print(df.columns.tolist())

print("\n── Data types & non-null counts ──────────────────────")
print(df.info())

print("\n── Summary statistics (numeric columns) ─────────────")
print(df.describe())

# ── 1.3  Quick data-quality snapshot ─────────────────────────
print("\n── Missing values per column ─────────────────────────")
missing = df.isnull().sum()
missing = missing[missing > 0]
print(missing if not missing.empty else "  None found ✅")

print("\n── Duplicate rows ────────────────────────────────────")
print(f"  {df.duplicated().sum()} duplicate(s) found")

print("\n── Unique values in key columns ──────────────────────")
for col in ["gender", "SeniorCitizen", "Partner", "Dependents",
            "Contract", "PaymentMethod", "Churn"]:
    if col in df.columns:
        print(f"  {col:20s}: {df[col].unique().tolist()}")


# ============================================================
# PHASE 2 — EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

print("\n" + "=" * 55)
print("  PHASE 2 — EXPLORATORY DATA ANALYSIS")
print("=" * 55)

# ── 2.0  Fix known data-quality issue ────────────────────────
#  TotalCharges is stored as object in the Kaggle dataset;
#  blank strings represent customers with 0 tenure.
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(0, inplace=True)

# Convert Churn to numeric for correlation later
df["Churn_Binary"] = (df["Churn"] == "Yes").astype(int)

print("\n✅ TotalCharges converted to numeric; blanks set to 0")

# ────────────────────────────────────────────────────────────
# PLOT 1 — Churn Distribution
# ────────────────────────────────────────────────────────────
churn_counts = df["Churn"].value_counts()
churn_pct = df["Churn"].value_counts(normalize=True) * 100

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].bar(churn_counts.index, churn_counts.values,
            color=["#4C9BE8", "#E8734C"], edgecolor="white", linewidth=1.5)
axes[0].set_title("Churn Count", fontweight="bold")
axes[0].set_xlabel("Churn"); axes[0].set_ylabel("Number of Customers")
for i, v in enumerate(churn_counts.values):
    axes[0].text(i, v + 30, str(v), ha="center", fontweight="bold")

axes[1].pie(churn_pct.values, labels=churn_pct.index,
            autopct="%1.1f%%", colors=["#4C9BE8", "#E8734C"],
            startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2})
axes[1].set_title("Churn Percentage", fontweight="bold")

plt.suptitle("Plot 1 — Churn Distribution", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("plot1_churn_distribution.png", bbox_inches="tight")
plt.show()
print("\n📊 Plot 1 saved: Churn Distribution")
print(f"   No Churn : {churn_pct['No']:.1f}%")
print(f"   Churn    : {churn_pct['Yes']:.1f}%  ← class imbalance (handle in Phase 3)")

# ────────────────────────────────────────────────────────────
# PLOT 2 — Churn vs Contract Type
# ────────────────────────────────────────────────────────────
contract_churn = (df.groupby(["Contract", "Churn"])
                    .size()
                    .reset_index(name="Count"))

plt.figure(figsize=(9, 4))
ax = sns.barplot(data=contract_churn, x="Contract", y="Count",
                 hue="Churn", palette={"No": "#4C9BE8", "Yes": "#E8734C"})
plt.title("Plot 2 — Churn vs Contract Type", fontweight="bold")
plt.xlabel("Contract Type"); plt.ylabel("Number of Customers")
plt.legend(title="Churn")
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}",
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig("plot2_contract_churn.png", bbox_inches="tight")
plt.show()
print("\n📊 Plot 2 saved: Churn vs Contract Type")
print("   Insight → Month-to-month contracts have the highest churn rate")

# ────────────────────────────────────────────────────────────
# PLOT 3 — Churn vs Tenure (Distribution)
# ────────────────────────────────────────────────────────────
plt.figure(figsize=(10, 4))
for label, color in [("No", "#4C9BE8"), ("Yes", "#E8734C")]:
    subset = df[df["Churn"] == label]["tenure"]
    plt.hist(subset, bins=30, alpha=0.65, label=f"Churn = {label}",
             color=color, edgecolor="white")

plt.title("Plot 3 — Tenure Distribution by Churn", fontweight="bold")
plt.xlabel("Tenure (months)"); plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.savefig("plot3_tenure_churn.png", bbox_inches="tight")
plt.show()
print("\n📊 Plot 3 saved: Tenure vs Churn")
print("   Insight → Customers who churn tend to have much lower tenure (new customers at risk)")

# ────────────────────────────────────────────────────────────
# PLOT 4 — Churn vs Monthly Charges
# ────────────────────────────────────────────────────────────
plt.figure(figsize=(10, 4))
for label, color in [("No", "#4C9BE8"), ("Yes", "#E8734C")]:
    subset = df[df["Churn"] == label]["MonthlyCharges"]
    plt.hist(subset, bins=30, alpha=0.65, label=f"Churn = {label}",
             color=color, edgecolor="white")

plt.title("Plot 4 — Monthly Charges Distribution by Churn", fontweight="bold")
plt.xlabel("Monthly Charges ($)"); plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.savefig("plot4_monthly_charges_churn.png", bbox_inches="tight")
plt.show()
print("\n📊 Plot 4 saved: Monthly Charges vs Churn")
print("   Insight → Higher monthly charges correlate with higher churn")

# ────────────────────────────────────────────────────────────
# PLOT 5 — Churn vs Categorical Features (2×2 grid)
# ────────────────────────────────────────────────────────────
cat_cols = ["gender", "SeniorCitizen", "Partner", "Dependents"]
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    ct = pd.crosstab(df[col], df["Churn"], normalize="index") * 100
    ct.plot(kind="bar", ax=axes[i], color=["#4C9BE8", "#E8734C"],
            edgecolor="white", linewidth=1)
    axes[i].set_title(f"Churn % by {col}", fontweight="bold")
    axes[i].set_ylabel("Percentage (%)"); axes[i].set_xlabel("")
    axes[i].legend(title="Churn")
    axes[i].tick_params(axis="x", rotation=0)

plt.suptitle("Plot 5 — Churn vs Demographic Features", fontsize=13,
             fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("plot5_demographic_churn.png", bbox_inches="tight")
plt.show()
print("\n📊 Plot 5 saved: Churn vs Demographic Features")
print("   Insight → Senior citizens churn ~2× more than non-seniors")
print("   Insight → Customers without partners/dependents churn more")

# ────────────────────────────────────────────────────────────
# PLOT 6 — Churn vs Internet & Phone Services
# ────────────────────────────────────────────────────────────
service_cols = ["PhoneService", "InternetService", "OnlineSecurity", "TechSupport"]
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
axes = axes.flatten()

for i, col in enumerate(service_cols):
    ct = pd.crosstab(df[col], df["Churn"], normalize="index") * 100
    ct.plot(kind="bar", ax=axes[i], color=["#4C9BE8", "#E8734C"],
            edgecolor="white", linewidth=1)
    axes[i].set_title(f"Churn % by {col}", fontweight="bold")
    axes[i].set_ylabel("Percentage (%)"); axes[i].set_xlabel("")
    axes[i].legend(title="Churn")
    axes[i].tick_params(axis="x", rotation=15)

plt.suptitle("Plot 6 — Churn vs Service Features", fontsize=13,
             fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("plot6_service_churn.png", bbox_inches="tight")
plt.show()
print("\n📊 Plot 6 saved: Churn vs Service Features")
print("   Insight → Fiber optic users churn the most among internet service types")
print("   Insight → Customers WITHOUT OnlineSecurity / TechSupport churn far more")

# ────────────────────────────────────────────────────────────
# PLOT 7 — Churn vs Payment Method
# ────────────────────────────────────────────────────────────
ct = pd.crosstab(df["PaymentMethod"], df["Churn"], normalize="index") * 100

plt.figure(figsize=(10, 4))
ax = ct.plot(kind="bar", color=["#4C9BE8", "#E8734C"],
             edgecolor="white", linewidth=1)
plt.title("Plot 7 — Churn % by Payment Method", fontweight="bold")
plt.ylabel("Percentage (%)"); plt.xlabel("")
plt.xticks(rotation=15, ha="right")
plt.legend(title="Churn")
plt.tight_layout()
plt.savefig("plot7_payment_churn.png", bbox_inches="tight")
plt.show()
print("\n📊 Plot 7 saved: Churn vs Payment Method")
print("   Insight → Electronic check users have the highest churn rate")

# ────────────────────────────────────────────────────────────
# PLOT 8 — Correlation Heatmap (numeric features)
# ────────────────────────────────────────────────────────────
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
corr = df[numeric_cols].corr()

plt.figure(figsize=(8, 5))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title("Plot 8 — Correlation Heatmap (Numeric Features)", fontweight="bold")
plt.tight_layout()
plt.savefig("plot8_correlation_heatmap.png", bbox_inches="tight")
plt.show()
print("\n📊 Plot 8 saved: Correlation Heatmap")
print("   Insight → Tenure is negatively correlated with Churn (longer = less likely to churn)")
print("   Insight → MonthlyCharges & TotalCharges are highly correlated (may drop one in Phase 3)")

# ────────────────────────────────────────────────────────────
# PHASE 2 SUMMARY
# ────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  ✅ PHASE 1 & 2 COMPLETE — SUMMARY")
print("=" * 55)
print(f"""
Dataset shape     : {df.shape[0]:,} rows × {df.shape[1]} columns
Churn rate        : {churn_pct['Yes']:.1f}% (class imbalance — handle with SMOTE in Phase 3)

Key findings:
  📌 Month-to-month contracts → highest churn risk
  📌 New customers (low tenure) → more likely to churn
  📌 Higher monthly charges → more likely to churn
  📌 Senior citizens → ~2× churn rate of non-seniors
  📌 No OnlineSecurity / TechSupport → much higher churn
  📌 Fiber optic internet users → highest churn in internet group
  📌 Electronic check payment → highest churn in payment group
  📌 MonthlyCharges & TotalCharges are highly correlated (consider dropping one)

Plots saved (8 files):
  plot1_churn_distribution.png
  plot2_contract_churn.png
  plot3_tenure_churn.png
  plot4_monthly_charges_churn.png
  plot5_demographic_churn.png
  plot6_service_churn.png
  plot7_payment_churn.png
  plot8_correlation_heatmap.png

Next → Phase 3: Feature Engineering & Preprocessing
""")
