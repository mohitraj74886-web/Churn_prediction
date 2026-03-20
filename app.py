"""
Telco Customer Churn Prediction API
====================================
Author : Mohit Raj
Model  : XGBoost (ROC-AUC=0.99, PR-AUC=0.98)
Stack  : FastAPI + joblib
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, validator

# ── Load model artifacts ───────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent

model          = joblib.load(BASE_DIR / "xgboost_churn_model.pkl")
feature_cols   = json.loads((BASE_DIR / "feature_columns.json").read_text())
threshold_data = json.loads((BASE_DIR / "threshold.json").read_text())
THRESHOLD      = threshold_data["threshold"]

# ── Retention recommendation map ──────────────────────────────────────────────
RETENTION_MAP = {
    "SatisfactionScore":       "Proactive CSAT outreach — personal callback from account manager",
    "TenureinMonths":          "Early loyalty program — offer 3-month discount at the 6-month mark",
    "Contract":                "Push annual contract — offer 15% discount for a 12-month sign-up",
    "MonthlyCharge":           "Price review — offer competitive rate match or bundle discount",
    "ChargePerService":        "Upsell add-ons — increase perceived value by bundling services",
    "TenureSegment":           "Onboarding intervention — assign a dedicated onboarding specialist",
    "Number_of_Referrals":     "Referral incentive — activate referral bonus to boost engagement",
    "StickyServiceCount":      "Bundle upsell — add online security or backup at no extra cost",
    "PremiumTechSupport":      "Offer free premium tech support trial for 3 months",
    "OnlineSecurity":          "Security bundle offer — highlight data protection benefits",
    "EngagementScore":         "Feature adoption campaign — demo streaming or security add-ons",
}
DEFAULT_ACTION = "General retention offer — $10 bill credit for next month"


# ── Pydantic input schema ──────────────────────────────────────────────────────
class CustomerInput(BaseModel):
    """
    Raw customer features — match what the retention team would know.
    All feature engineering is handled server-side.
    """
    # Demographics
    Age:                            int   = Field(..., ge=18, le=100,  example=45)
    Gender:                         str   = Field(...,                  example="Male")
    SeniorCitizen:                  str   = Field(...,                  example="No")
    Married:                        str   = Field(...,                  example="Yes")
    Dependents:                     str   = Field(...,                  example="No")
    NumberofDependents:             int   = Field(..., ge=0, le=10,     example=0)
    Under30:                        str   = Field(...,                  example="No")

    # Account info
    TenureinMonths:                 int   = Field(..., ge=0, le=120,    example=12)
    Contract:                       str   = Field(...,                  example="Month-to-Month")
    PaperlessBilling:               str   = Field(...,                  example="Yes")
    PaymentMethod:                  str   = Field(...,                  example="Bank Withdrawal")
    MonthlyCharge:                  float = Field(..., ge=0,            example=79.85)
    TotalCharges:                   float = Field(..., ge=0,            example=958.20)
    TotalRevenue:                   float = Field(..., ge=0,            example=1024.10)
    TotalRefunds:                   float = Field(0.0, ge=0,            example=0.0)
    TotalExtraDataCharges:          int   = Field(0,   ge=0,            example=0)
    TotalLongDistanceCharges:       float = Field(0.0, ge=0,            example=200.0)
    CLTV:                           int   = Field(..., ge=0,            example=5000)

    # Services
    PhoneService:                   str   = Field(...,                  example="Yes")
    MultipleLines:                  str   = Field(...,                  example="No")
    InternetService:                str   = Field(...,                  example="Yes")
    InternetType:                   str   = Field(...,                  example="Fiber Optic")
    AvgMonthlyGBDownload:           int   = Field(0,   ge=0,            example=17)
    OnlineSecurity:                 str   = Field(...,                  example="No")
    OnlineBackup:                   str   = Field(...,                  example="No")
    DeviceProtectionPlan:           str   = Field(...,                  example="No")
    PremiumTechSupport:             str   = Field(...,                  example="No")
    StreamingTV:                    str   = Field(...,                  example="Yes")
    StreamingMovies:                str   = Field(...,                  example="Yes")
    StreamingMusic:                 str   = Field(...,                  example="No")
    UnlimitedData:                  str   = Field(...,                  example="Yes")
    AvgMonthlyLongDistanceCharges:  float = Field(0.0, ge=0,            example=25.0)

    # Engagement
    ReferredaFriend:                str   = Field(...,                  example="No")
    Number_of_Referrals:            int   = Field(0,   ge=0,            example=0)
    Offer:                          Optional[str] = Field(None,         example="None")
    SatisfactionScore:              int   = Field(..., ge=1, le=5,      example=2)

    @validator("Gender")
    def validate_gender(cls, v):
        if v not in ("Male", "Female"):
            raise ValueError("Gender must be 'Male' or 'Female'")
        return v

    @validator("Contract")
    def validate_contract(cls, v):
        valid = ("Month-to-Month", "One Year", "Two Year")
        if v not in valid:
            raise ValueError(f"Contract must be one of {valid}")
        return v

    @validator("InternetType")
    def validate_internet_type(cls, v):
        valid = ("Fiber Optic", "DSL", "Cable", "No Internet")
        if v not in valid:
            raise ValueError(f"InternetType must be one of {valid}")
        return v


# ── Response schema ────────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    churn_probability:    float
    churn_prediction:     bool
    risk_level:           str
    threshold_used:       float
    priority_score:       float
    top_churn_driver:     str
    retention_action:     str
    clv_at_risk:          int
    model_version:        str = "XGBoost-v1.0"


# ── Feature engineering (mirrors EDA notebook) ─────────────────────────────────
def engineer_features(data: dict) -> pd.DataFrame:
    """Replicate all feature engineering from the EDA/training notebook."""
    yn = {"Yes": 1, "No": 0}

    binary_cols = [
        "Under30", "SeniorCitizen", "Married", "Dependents",
        "ReferredaFriend", "PhoneService", "MultipleLines",
        "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtectionPlan", "PremiumTechSupport",
        "StreamingTV", "StreamingMovies", "StreamingMusic",
        "UnlimitedData", "PaperlessBilling"
    ]

    row = {}

    # Binary encodings
    for col in binary_cols:
        row[col] = yn.get(data.get(col, "No"), 0)

    # Gender
    row["Gender"] = 1 if data["Gender"] == "Male" else 0

    # Numeric passthrough
    for col in [
        "Age", "NumberofDependents", "TenureinMonths",
        "AvgMonthlyLongDistanceCharges", "AvgMonthlyGBDownload",
        "MonthlyCharge", "TotalCharges", "TotalRefunds",
        "TotalExtraDataCharges", "TotalLongDistanceCharges",
        "TotalRevenue", "SatisfactionScore", "CLTV",
        "Number_of_Referrals"
    ]:
        row[col] = data.get(col, 0)

    # Contract ordinal
    contract_map = {"Month-to-Month": 0, "One Year": 1, "Two Year": 2}
    row["Contract"] = contract_map.get(data["Contract"], 0)

    # One-hot: PaymentMethod
    pm_options = ["Bank Withdrawal", "Credit Card", "Mailed Check"]
    for opt in pm_options:
        row[f"PaymentMethod_{opt}"] = int(data.get("PaymentMethod") == opt)

    # One-hot: InternetType
    it_options = ["Cable", "DSL", "Fiber Optic", "No Internet"]
    for opt in it_options:
        row[f"InternetType_{opt}"] = int(data.get("InternetType", "No Internet") == opt)

    # One-hot: Offer
    offer_val = data.get("Offer") or "None"
    offer_options = ["None", "Offer A", "Offer B", "Offer C", "Offer D", "Offer E"]
    for opt in offer_options:
        row[f"Offer_{opt}"] = int(offer_val == opt)

    # ── Engineered features ─────────────────────────────────────────────────
    tenure = data["TenureinMonths"]

    # Tenure segment (0=new, 4=loyal)
    if   tenure <= 12: row["TenureSegment"] = 0
    elif tenure <= 24: row["TenureSegment"] = 1
    elif tenure <= 36: row["TenureSegment"] = 2
    elif tenure <= 48: row["TenureSegment"] = 3
    else:              row["TenureSegment"] = 4

    # Service counts
    service_cols = [
        "PhoneService", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtectionPlan", "PremiumTechSupport", "StreamingTV",
        "StreamingMovies", "StreamingMusic", "UnlimitedData", "MultipleLines"
    ]
    row["ServiceCount"] = sum(row.get(c, 0) for c in service_cols)

    sticky = ["OnlineSecurity", "OnlineBackup", "DeviceProtectionPlan", "PremiumTechSupport"]
    streaming = ["StreamingTV", "StreamingMovies", "StreamingMusic"]
    row["StickyServiceCount"]    = sum(row.get(c, 0) for c in sticky)
    row["StreamingServiceCount"] = sum(row.get(c, 0) for c in streaming)

    # Charge per service
    row["ChargePerService"] = data["MonthlyCharge"] / (row["ServiceCount"] + 1)

    # Revenue risk (normalized — use 120 as max monthly charge proxy)
    row["RevenueRisk"] = data["CLTV"] * (data["MonthlyCharge"] / 120)

    # Engagement score (normalized 0-1, approximate)
    raw_engage = (
        row["ReferredaFriend"] * 2 +
        row["StreamingServiceCount"] +
        (data.get("AvgMonthlyGBDownload", 0) / 50)   # 50 GB as rough max
    )
    row["EngagementScore"] = min(raw_engage / 6, 1.0)  # cap at 1.0

    # Long distance share
    row["LongDistShare"] = data.get("TotalLongDistanceCharges", 0) / (data["TotalRevenue"] + 1)

    # ── Align with training columns ─────────────────────────────────────────
    df = pd.DataFrame([row])

    # Add any missing columns with 0
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    # Drop RevenueRisk — excluded from model input (kept for priority score)
    revenue_risk = df["RevenueRisk"].values[0] if "RevenueRisk" in df.columns else 0
    df_model = df[feature_cols].copy()

    return df_model, revenue_risk


def get_top_shap_driver(df_input: pd.DataFrame) -> str:
    """Identify the most influential feature pushing toward churn."""
    try:
        import shap
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df_input)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        shap_series = pd.Series(shap_values[0], index=df_input.columns)
        pos_shap    = shap_series[shap_series > 0].sort_values(ascending=False)
        for feat in pos_shap.index:
            if feat in RETENTION_MAP:
                return feat
        return pos_shap.index[0] if len(pos_shap) > 0 else "Unknown"
    except Exception:
        # Fallback if SHAP fails — use model feature importances
        importances = pd.Series(model.feature_importances_, index=feature_cols)
        return importances.nlargest(1).index[0]


# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Churn Prediction API",
    description=(
        "XGBoost model predicting customer churn probability with SHAP-driven "
        "retention recommendations. ROC-AUC=0.9917, PR-AUC=0.9812."
    ),
    version="1.0.0",
    contact={"name": "Mohit Raj", "url": "https://github.com/mohitraj74886-web"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, tags=["Info"])
def root():
    """Landing page with API overview and quick links."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
      <title>Churn Prediction API</title>
      <style>
        body { font-family: 'Segoe UI', sans-serif; max-width: 720px; margin: 60px auto;
               padding: 0 24px; background: #0f0f0f; color: #e0e0e0; }
        h1   { font-size: 28px; font-weight: 600; color: #fff; margin-bottom: 4px; }
        p    { color: #888; line-height: 1.7; }
        .badge { display: inline-block; background: #1a1a1a; border: 1px solid #333;
                 border-radius: 6px; padding: 3px 10px; font-size: 12px;
                 color: #4ade80; margin-right: 6px; }
        a    { color: #6366f1; text-decoration: none; }
        a:hover { text-decoration: underline; }
        .card { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 10px;
                padding: 20px 24px; margin: 20px 0; }
        code { background: #2a2a2a; padding: 2px 7px; border-radius: 4px;
               font-size: 13px; color: #f472b6; }
      </style>
    </head>
    <body>
      <h1>Churn Prediction API</h1>
      <p>
        <span class="badge">XGBoost v1.0</span>
        <span class="badge">ROC-AUC 0.9917</span>
        <span class="badge">PR-AUC 0.9812</span>
      </p>
      <p>Predicts customer churn probability with SHAP-driven retention recommendations.</p>
      <div class="card">
        <b>Endpoints</b><br><br>
        <code>POST /predict</code> — predict churn for a single customer<br><br>
        <code>POST /predict/batch</code> — predict for up to 100 customers<br><br>
        <code>GET  /health</code> — health check<br><br>
        <code>GET  /model/info</code> — model metadata & feature list
      </div>
      <div class="card">
        <b>Quick links</b><br><br>
        <a href="/docs">Interactive API docs (Swagger UI)</a><br><br>
        <a href="/redoc">ReDoc documentation</a><br><br>
        <a href="/health">Health check</a>
      </div>
    </body>
    </html>
    """


@app.get("/health", tags=["Info"])
def health():
    """Health check endpoint — used by Render to verify the service is running."""
    return {
        "status":    "healthy",
        "model":     "XGBoost-v1.0",
        "threshold": THRESHOLD,
        "features":  len(feature_cols)
    }


@app.get("/model/info", tags=["Info"])
def model_info():
    """Returns model metadata, performance metrics, and feature list."""
    return {
        "model_type":       "XGBClassifier",
        "version":          "1.0.0",
        "threshold":        THRESHOLD,
        "n_features":       len(feature_cols),
        "feature_names":    feature_cols,
        "cv_metrics": {
            "roc_auc":  "0.9923 ± 0.0023",
            "pr_auc":   "0.9828 ± 0.0042",
            "f1":       "0.9241 ± 0.0069",
            "recall":   "0.9211 ± 0.0146",
            "precision":"0.9274 ± 0.0102"
        },
        "test_metrics": {
            "roc_auc":   0.9917,
            "pr_auc":    0.9812,
            "f1":        0.9254,
            "precision": 0.9394,
            "recall":    0.9118
        },
        "training_dataset": "IBM Telco Customer Churn (7,043 rows × 55 features)",
        "author":           "Mohit Raj"
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(customer: CustomerInput):
    """
    Predict churn probability for a single customer.

    Returns:
    - **churn_probability**: probability of churning (0–1)
    - **churn_prediction**: True if above threshold
    - **risk_level**: Low / Medium / High / Critical
    - **priority_score**: CLV-weighted urgency score (0–100)
    - **top_churn_driver**: the feature most responsible for churn risk
    - **retention_action**: recommended business intervention
    """
    try:
        data        = customer.dict()
        df_input, revenue_risk = engineer_features(data)

        # Model prediction
        prob        = float(model.predict_proba(df_input)[0][1])
        prediction  = prob >= THRESHOLD

        # Risk level
        if prob < 0.30:   risk = "Low"
        elif prob < 0.55: risk = "Medium"
        elif prob < 0.75: risk = "High"
        else:             risk = "Critical"

        # CLV-weighted priority score (0–100)
        cltv_norm      = min(data["CLTV"] / 8000, 1.0)   # 8000 as rough max CLTV
        priority_score = round(np.sqrt(prob * cltv_norm) * 100, 1)

        # SHAP driver & retention action
        top_driver     = get_top_shap_driver(df_input)
        action         = RETENTION_MAP.get(top_driver, DEFAULT_ACTION)

        return PredictionResponse(
            churn_probability = round(prob, 4),
            churn_prediction  = bool(prediction),
            risk_level        = risk,
            threshold_used    = THRESHOLD,
            priority_score    = priority_score,
            top_churn_driver  = top_driver,
            retention_action  = action,
            clv_at_risk       = data["CLTV"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(customers: list[CustomerInput]):
    """
    Predict churn for a batch of up to 100 customers.
    Returns a list of predictions sorted by priority score (highest first).
    """
    if len(customers) > 100:
        raise HTTPException(
            status_code=422,
            detail="Batch size limited to 100 customers per request."
        )

    results = []
    for i, customer in enumerate(customers):
        try:
            pred = predict(customer)
            results.append({"customer_index": i, **pred.dict()})
        except Exception as e:
            results.append({"customer_index": i, "error": str(e)})

    # Sort by priority score descending
    results.sort(key=lambda x: x.get("priority_score", 0), reverse=True)
    return {
        "total":           len(results),
        "at_risk":         sum(1 for r in results if r.get("churn_prediction")),
        "predictions":     results
    }
