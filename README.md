# Telco Customer Churn Prediction API

**XGBoost model** predicting customer churn probability with SHAP-driven retention recommendations.

| Metric | Score |
|--------|-------|
| ROC-AUC (test) | 0.9917 |
| PR-AUC (test) | 0.9812 |
| F1 (test) | 0.9254 |
| CV ROC-AUC | 0.9923 ± 0.0023 |
| Training rows | 7,043 |
| Features | 55 (+ engineered) |

---

## Project structure

```
churn-predictor/
├── app.py                      ← FastAPI application
├── requirements.txt            ← Python dependencies
├── xgboost_churn_model.pkl     ← Trained model (download from Colab)
├── feature_columns.json        ← Feature order used during training
├── threshold.json              ← Optimal classification threshold
├── notebooks/
│   ├── telco_churn_eda.ipynb
│   └── telco_churn_modelling.ipynb
└── README.md
```

---

## Step 1 — Download model files from Colab

Add this cell at the **end** of your modelling notebook and run it:

```python
import joblib, json
from google.colab import files

# Save model
joblib.dump(trained_models['XGBoost'], 'xgboost_churn_model.pkl')

# Save feature columns
with open('feature_columns.json', 'w') as f:
    json.dump(X_train.columns.tolist(), f)

# Save threshold
with open('threshold.json', 'w') as f:
    json.dump({'threshold': float(OPTIMAL_THRESHOLD)}, f)

# Download all three files to your laptop
files.download('xgboost_churn_model.pkl')
files.download('feature_columns.json')
files.download('threshold.json')
```

Place the three downloaded files in the `churn-predictor/` folder alongside `app.py`.

---

## Step 2 — Run locally (VSCode / terminal)

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the server
uvicorn app:app --reload --port 8000
```

Open http://localhost:8000 — you'll see the landing page.
Open http://localhost:8000/docs — interactive Swagger UI to test all endpoints.

---

## Step 3 — Test the API locally

### Single prediction (curl)

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 45,
    "Gender": "Male",
    "SeniorCitizen": "No",
    "Married": "No",
    "Dependents": "No",
    "NumberofDependents": 0,
    "Under30": "No",
    "TenureinMonths": 3,
    "Contract": "Month-to-Month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Bank Withdrawal",
    "MonthlyCharge": 89.50,
    "TotalCharges": 268.50,
    "TotalRevenue": 268.50,
    "TotalRefunds": 0,
    "TotalExtraDataCharges": 0,
    "TotalLongDistanceCharges": 0,
    "CLTV": 5200,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Yes",
    "InternetType": "Fiber Optic",
    "AvgMonthlyGBDownload": 20,
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtectionPlan": "No",
    "PremiumTechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "StreamingMusic": "No",
    "UnlimitedData": "Yes",
    "AvgMonthlyLongDistanceCharges": 0,
    "ReferredaFriend": "No",
    "Number_of_Referrals": 0,
    "Offer": "None",
    "SatisfactionScore": 2
  }'
```

### Expected response

```json
{
  "churn_probability": 0.8731,
  "churn_prediction": true,
  "risk_level": "Critical",
  "threshold_used": 0.5,
  "priority_score": 67.4,
  "top_churn_driver": "SatisfactionScore",
  "retention_action": "Proactive CSAT outreach — personal callback from account manager",
  "clv_at_risk": 5200,
  "model_version": "XGBoost-v1.0"
}
```

---

## Step 4 — Deploy on Render (free, no credit card needed)

### 4a. Push to GitHub

```bash
# In the churn-predictor/ folder
git init
git add .
git commit -m "Initial commit — churn prediction API"

# Create a new repo on github.com, then:
git remote add origin https://github.com/YOUR_USERNAME/churn-predictor.git
git branch -M main
git push -u origin main
```

> **Important:** Add `*.pkl` to `.gitignore` if your model file is over 100MB.
> If it is, use Git LFS: `git lfs track "*.pkl"` then `git add .gitattributes`

### 4b. Create Web Service on Render

1. Go to **render.com** → Sign up with GitHub (free)
2. Click **New → Web Service**
3. Connect your GitHub repo (`churn-predictor`)
4. Fill in the settings:

| Setting | Value |
|---------|-------|
| **Name** | `churn-predictor` (or any name) |
| **Region** | Singapore (closest to India) |
| **Branch** | `main` |
| **Runtime** | `Python 3` |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `uvicorn app:app --host 0.0.0.0 --port $PORT` |
| **Instance Type** | `Free` |

5. Click **Create Web Service**
6. Wait ~3 minutes for the build to complete
7. Your API is live at: `https://churn-predictor.onrender.com`

> **Free tier note:** The service sleeps after 15 minutes of inactivity.
> First request after sleep takes ~30 seconds to wake up. This is fine for a portfolio project.

### 4c. Verify deployment

```bash
# Health check
curl https://churn-predictor.onrender.com/health

# Swagger UI
open https://churn-predictor.onrender.com/docs
```

---

## Step 5 — What to put on your resume

```
Customer Churn Prediction API
Python · XGBoost · FastAPI · SHAP · Render

Built end-to-end churn prediction system on IBM Telco data (7,043 customers,
55 features). XGBoost with 5-fold CV: ROC-AUC=0.99, PR-AUC=0.98. Business-
optimal threshold via cost matrix. SHAP-driven retention recommendations per
customer. Deployed as a live REST API on Render.

Live API:  https://churn-predictor.onrender.com
Docs:      https://churn-predictor.onrender.com/docs
GitHub:    https://github.com/YOUR_USERNAME/churn-predictor
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Landing page |
| GET | `/health` | Health check |
| GET | `/model/info` | Model metadata & CV metrics |
| POST | `/predict` | Single customer prediction |
| POST | `/predict/batch` | Batch prediction (up to 100) |

---

## Tech stack

- **Model:** XGBoost (trained in Google Colab)
- **API:** FastAPI + Pydantic v2
- **Explainability:** SHAP TreeExplainer
- **Hosting:** Render.com (free tier)
- **Serialization:** joblib
