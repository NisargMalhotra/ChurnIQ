# 📉 ChurnIQ — Customer Churn Intelligence Platform

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?style=flat&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18.2+-61DAFB?style=flat&logo=react&logoColor=black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-FF6600?style=flat)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-8A2BE2?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

> An end-to-end machine learning web application that predicts which customers are likely to cancel their subscription — built with FastAPI, React, XGBoost, LightGBM, and SHAP explainability.

---

## 🖥️ Live Demo

![ChurnIQ Dashboard](assets/dashboard_preview.png)

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [ML Pipeline](#-ml-pipeline)
- [Getting Started](#-getting-started)
- [How to Use the App](#-how-to-use-the-app)
- [Model Performance](#-model-performance)
- [Key Insights](#-key-insights-from-shap)
- [API Reference](#-api-reference)
- [Dataset](#-dataset)
- [License](#-license)

---

## 🔍 Overview

ChurnIQ is a full-stack machine learning application designed for telecom/subscription companies to identify customers at risk of churning. Upload a customer dataset, and the platform instantly predicts churn probability for every customer, ranks them by risk level, and shows which factors are driving churn — all through a clean, interactive dashboard.

**Two modes:**
- **⚡ Predict** — Upload a dataset and get instant predictions using the pre-trained model
- **🔁 Retrain + Predict** — Upload a labeled dataset to retrain all models from scratch, auto-select the best one, then predict

---

## ✨ Features

- 📂 **Drag & drop CSV upload** — works with any Telco-format customer dataset
- 🤖 **3 ML models compared** — Logistic Regression, XGBoost, LightGBM
- 🏆 **Auto model selection** — best model chosen by ROC-AUC score
- 📊 **4 live charts** — risk donut, churn bar, probability histogram, SHAP feature importance
- 🧠 **SHAP explainability** — understand *why* a customer is predicted to churn
- 👥 **Customer table** — every customer ranked by churn probability with risk badge
- 🔍 **Search & filter** — filter by High / Medium / Low risk, search by customer ID
- 💰 **Revenue at risk** — total monthly charges of customers predicted to churn
- 📄 **Pagination** — handles thousands of customers cleanly

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | React 18, Vite, react-dropzone |
| **Backend** | FastAPI, Uvicorn |
| **ML Models** | Scikit-learn, XGBoost, LightGBM |
| **Explainability** | SHAP (TreeExplainer / LinearExplainer) |
| **Data** | Pandas, NumPy |
| **Imbalance Handling** | SMOTE (imbalanced-learn) |
| **Visualization** | Matplotlib, Seaborn |
| **Serialization** | Joblib |

---

## 📁 Project Structure

```
churniq/
│
├── 📂 backend/                   # FastAPI server
│   ├── main.py                   # API routes & ML logic
│   ├── requirements.txt          # Python dependencies
│   ├── best_model.pkl            # Trained model (generated)
│   ├── scaler.pkl                # Feature scaler (generated)
│   ├── feature_names.pkl         # Feature list (generated)
│   └── best_model_name.pkl       # Best model name (generated)
│
├── 📂 frontend/                  # React app
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── main.jsx
│       └── App.jsx               # Full UI
│
├── 📂 notebooks/                 # ML pipeline (Jupyter-style .py files)
│   ├── churn_phase1_2.py         # EDA & visualization
│   ├── churn_phase3.py           # Feature engineering & preprocessing
│   ├── churn_phase4.py           # Model training & comparison
│   └── churn_phase5.py           # SHAP explainability
│
├── 📂 assets/                    # Screenshots for README
│   └── dashboard_preview.png
│
├── customer_churn.csv            # Dataset (not committed — see Dataset section)
├── .gitignore
└── README.md
```

---

## 🧠 ML Pipeline

```
Raw CSV
   │
   ▼
Phase 1-2: EDA & Visualization
   │   • Churn distribution, contract types, tenure, charges
   │   • 8 exploratory plots saved as PNG
   │
   ▼
Phase 3: Feature Engineering & Preprocessing
   │   • Fix TotalCharges (blank → 0)
   │   • New features: ChargesPerMonth, TenureGroup, NumServices, HighValueCustomer
   │   • Label encode binary cols, One-hot encode multi-class cols
   │   • StandardScaler on numeric features (no data leakage)
   │   • SMOTE to fix class imbalance (27% → 50%)
   │
   ▼
Phase 4: Model Training & Comparison
   │   • Logistic Regression (baseline)
   │   • XGBoost (gradient boosting)
   │   • LightGBM (fast boosting)
   │   • Compare: Accuracy, Precision, Recall, F1, ROC-AUC
   │   • Auto-select best model by ROC-AUC
   │
   ▼
Phase 5: SHAP Explainability
   │   • Global feature importance (bar + dot plots)
   │   • Per-customer waterfall plot
   │   • Dependence plots: Tenure & MonthlyCharges
   │
   ▼
FastAPI Backend → React Frontend Dashboard
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+ (LTS)
- Git

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/churniq.git
cd churniq
```

### 2. Train the model (run once)

Place `customer_churn.csv` in the root folder, then run the pipeline in order:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm imbalanced-learn shap joblib
```

```bash
python notebooks/churn_phase1_2.py   # EDA
python notebooks/churn_phase3.py     # Preprocessing
python notebooks/churn_phase4.py     # Model training
python notebooks/churn_phase5.py     # SHAP
```

### 3. Copy model files to backend

```bash
# Windows
copy *.pkl backend\

# Mac / Linux
cp *.pkl backend/
```

### 4. Start the backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

Backend runs at: **http://127.0.0.1:8000**

### 5. Start the frontend

Open a **new terminal**:

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at: **http://localhost:5173**

---

## 📖 How to Use the App

1. Open **http://localhost:5173** in your browser
2. Choose a mode:
   - **⚡ Predict** — uses the pre-trained model (faster)
   - **🔁 Retrain + Predict** — retrains on your data first (dataset must have a `Churn` column)
3. Drag & drop your `.csv` file or click to browse
4. Click **Run Prediction**
5. View:
   - Summary cards (total customers, churn count, revenue at risk)
   - 4 charts (risk distribution, churn vs stay, probability histogram, SHAP)
   - Full customer table sorted by churn probability
   - Filter by **High / Medium / Low** risk or search by customer ID

---

## 📈 Model Performance

Results on the Telco Customer Churn dataset (test set, 20% holdout):

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | ~80% | ~65% | ~56% | ~60% | ~84% |
| XGBoost | ~81% | ~67% | ~57% | ~62% | ~85% |
| **LightGBM** | **~82%** | **~68%** | **~59%** | **~63%** | **~86%** |

> **Why ROC-AUC?** For churn prediction, we care about *ranking* customers by risk. A high AUC means the model correctly separates churners from non-churners even with class imbalance.

> **Why Recall matters too?** Missing a churner (False Negative) is more costly than a false alarm — the business permanently loses that customer.

---

## 🔬 Key Insights from SHAP

After training, SHAP analysis revealed the following business insights:

| # | Insight | Business Action |
|---|---|---|
| 1 | **Low tenure = highest churn risk** | Engage new customers in months 1–12 with onboarding offers |
| 2 | **Month-to-month contracts churn the most** | Offer discounts to move customers to annual plans |
| 3 | **High monthly charges + low tenure = danger zone** | Introduce loyalty pricing for new high-value customers |
| 4 | **No OnlineSecurity / TechSupport = more churn** | Bundle these into starter plans |
| 5 | **Electronic check users churn more** | Incentivize auto-pay via credit card or bank transfer |
| 6 | **Fiber optic internet users churn more** | Investigate service quality issues with fiber customers |

---

## 🔌 API Reference

Base URL: `http://127.0.0.1:8000`

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `GET` | `/model-status` | Check if model is loaded |
| `POST` | `/predict` | Predict churn on uploaded CSV |
| `POST` | `/retrain` | Retrain models on labeled CSV, then predict |

### POST `/predict`
- **Body:** `multipart/form-data` with `file` (CSV)
- **Returns:** Summary stats, per-customer predictions, base64-encoded charts

### POST `/retrain`
- **Body:** `multipart/form-data` with `file` (CSV — must have `Churn` column)
- **Returns:** Training results for all 3 models, best model name, ROC-AUC

---

## 📦 Dataset

This project uses the **Telco Customer Churn** dataset from Kaggle.

👉 [Download here](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

The dataset contains 7,043 customers with 21 features including:
- Demographics: gender, SeniorCitizen, Partner, Dependents
- Services: PhoneService, InternetService, OnlineSecurity, StreamingTV, etc.
- Account: tenure, Contract, PaymentMethod, MonthlyCharges, TotalCharges
- Target: `Churn` (Yes / No)

> ⚠️ `customer_churn.csv` is not committed to this repo. Download it from Kaggle and place it in the root folder before running the pipeline.

---

## 🙏 Acknowledgements

- [Kaggle Telco Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) by IBM
- [SHAP](https://github.com/slundberg/shap) by Scott Lundberg
- [FastAPI](https://fastapi.tiangolo.com/) by Sebastián Ramírez

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  Built with ❤️ as a portfolio ML project
</div>
