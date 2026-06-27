from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import joblib
import pandas as pd
import numpy as np
import os

# ── Load model ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, '../models/loan_default_model.pkl'))
categorical_cols = joblib.load(os.path.join(BASE_DIR, '../models/categorical_cols.pkl'))
numerical_cols = joblib.load(os.path.join(BASE_DIR, '../models/numerical_cols.pkl'))

app = FastAPI(
    title="Loan Default Risk Scoring API",
    description="Predicts whether a loan applicant will default using LightGBM, enriched with bureau and previous application credit history",
    version="2.0.0"
)

# ── Input schema ──────────────────────────────────────────────
class LoanInput(BaseModel):
    NAME_CONTRACT_TYPE: str
    CODE_GENDER: str
    FLAG_OWN_CAR: str
    FLAG_OWN_REALTY: str
    CNT_CHILDREN: int
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: float
    AMT_GOODS_PRICE: float
    NAME_TYPE_SUITE: str = "Unaccompanied"
    NAME_INCOME_TYPE: str
    NAME_EDUCATION_TYPE: str
    NAME_FAMILY_STATUS: str
    NAME_HOUSING_TYPE: str
    REGION_POPULATION_RELATIVE: float
    DAYS_REGISTRATION: float
    OWN_CAR_AGE: float = 0.0
    FLAG_MOBIL: int = 1
    FLAG_EMP_PHONE: int
    FLAG_WORK_PHONE: int
    FLAG_CONT_MOBILE: int = 1
    FLAG_PHONE: int
    FLAG_EMAIL: int
    OCCUPATION_TYPE: str = "Laborers"
    CNT_FAM_MEMBERS: float
    REGION_RATING_CLIENT: int
    REGION_RATING_CLIENT_W_CITY: int
    WEEKDAY_APPR_PROCESS_START: str
    HOUR_APPR_PROCESS_START: int
    REG_REGION_NOT_LIVE_REGION: int = 0
    REG_REGION_NOT_WORK_REGION: int = 0
    LIVE_REGION_NOT_WORK_REGION: int = 0
    REG_CITY_NOT_LIVE_CITY: int
    REG_CITY_NOT_WORK_CITY: int
    LIVE_CITY_NOT_WORK_CITY: int
    ORGANIZATION_TYPE: str
    EXT_SOURCE_2: float
    EXT_SOURCE_3: float
    DEF_30_CNT_SOCIAL_CIRCLE: float = 0.0
    DEF_60_CNT_SOCIAL_CIRCLE: float = 0.0
    DAYS_LAST_PHONE_CHANGE: float
    FLAG_DOCUMENT_3: int
    FLAG_DOCUMENT_6: int = 0
    FLAG_DOCUMENT_8: int = 0
    FLAG_DOCUMENT_2: int = 0
    FLAG_DOCUMENT_4: int = 0
    FLAG_DOCUMENT_5: int = 0
    FLAG_DOCUMENT_7: int = 0
    FLAG_DOCUMENT_9: int = 0
    FLAG_DOCUMENT_10: int = 0
    FLAG_DOCUMENT_11: int = 0
    FLAG_DOCUMENT_12: int = 0
    FLAG_DOCUMENT_13: int = 0
    FLAG_DOCUMENT_14: int = 0
    FLAG_DOCUMENT_15: int = 0
    FLAG_DOCUMENT_16: int = 0
    FLAG_DOCUMENT_17: int = 0
    FLAG_DOCUMENT_18: int = 0
    FLAG_DOCUMENT_19: int = 0
    FLAG_DOCUMENT_20: int = 0
    FLAG_DOCUMENT_21: int = 0
    OBS_30_CNT_SOCIAL_CIRCLE: float = 0.0
    OBS_60_CNT_SOCIAL_CIRCLE: float = 0.0
    DAYS_ID_PUBLISH: float = -1000.0
    AMT_REQ_CREDIT_BUREAU_HOUR: float = 0.0
    AMT_REQ_CREDIT_BUREAU_DAY: float = 0.0
    AMT_REQ_CREDIT_BUREAU_WEEK: float = 0.0
    AMT_REQ_CREDIT_BUREAU_MON: float = 0.0
    AMT_REQ_CREDIT_BUREAU_QRT: float = 0.0
    AMT_REQ_CREDIT_BUREAU_YEAR: float = 0.0
    AGE_YEARS: float
    YEARS_EMPLOYED: float
    CREDIT_INCOME_RATIO: float
    ANNUITY_INCOME_RATIO: float
    CREDIT_GOODS_RATIO: float
    INCOME_PER_PERSON: float

    # ── Bureau history features — optional, default = no bureau history ──
    BUREAU_LOAN_COUNT: float = 0.0
    BUREAU_ACTIVE_LOANS: float = 0.0
    BUREAU_CREDIT_TYPES: float = 0.0
    BUREAU_DAYS_CREDIT_MEAN: Optional[float] = None
    BUREAU_DAYS_CREDIT_MIN: Optional[float] = None
    BUREAU_MAX_OVERDUE_DAYS: float = 0.0
    BUREAU_MAX_AMT_OVERDUE: float = 0.0
    BUREAU_TOTAL_PROLONG: float = 0.0
    BUREAU_TOTAL_CREDIT_SUM: float = 0.0
    BUREAU_TOTAL_DEBT: float = 0.0
    BUREAU_TOTAL_OVERDUE: float = 0.0
    BUREAU_DEBT_CREDIT_RATIO: float = 0.0

    # ── Previous Home Credit application features — optional, default = no prior application ──
    PREV_APP_COUNT: float = 0.0
    PREV_APPROVED_COUNT: float = 0.0
    PREV_REFUSED_COUNT: float = 0.0
    PREV_CANCELLED_COUNT: float = 0.0
    PREV_DAYS_DECISION_MEAN: Optional[float] = None
    PREV_DAYS_DECISION_MIN: Optional[float] = None
    PREV_AMT_APPLICATION_MEAN: Optional[float] = None
    PREV_AMT_CREDIT_MEAN: Optional[float] = None
    PREV_AMT_ANNUITY_MEAN: Optional[float] = None
    PREV_CNT_PAYMENT_MEAN: Optional[float] = None
    PREV_RATE_DOWN_PAYMENT_MEAN: Optional[float] = None
    PREV_CREDIT_APPLICATION_RATIO: Optional[float] = None
    PREV_REFUSAL_RATE: float = 0.0


# ── Endpoints ─────────────────────────────────────────────────
@app.get("/")
def home():
    return {
        "message": "Loan Default Risk Scoring API is running!",
        "version": "2.0.0 — enriched with bureau and previous application credit history",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model": "LightGBM Loan Default Predictor (enriched, ROC-AUC 0.770)"}

@app.post("/predict")
def predict(loan: LoanInput):
    input_dict = loan.dict()
    df = pd.DataFrame([input_dict])

    # Auto-compute "no history" flags based on whether bureau/previous application
    # data was actually supplied. If the caller didn't provide real bureau data,
    # BUREAU_LOAN_COUNT stays at its default of 0, which we interpret as no history.
    df['NO_BUREAU_HISTORY'] = (df['BUREAU_LOAN_COUNT'] == 0).astype(int)
    df['NO_PREV_APPLICATION'] = (df['PREV_APP_COUNT'] == 0).astype(int)

    # Fix categorical types
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Predict
    default_probability = model.predict_proba(df)[0][1]
    prediction = model.predict(df)[0]

    # Risk tier
    if default_probability >= 0.75:
        risk_tier = "D — VERY HIGH RISK"
    elif default_probability >= 0.50:
        risk_tier = "C — HIGH RISK"
    elif default_probability >= 0.25:
        risk_tier = "B — MEDIUM RISK"
    else:
        risk_tier = "A — LOW RISK"

    return {
        "default_predicted": bool(prediction),
        "default_probability": round(float(default_probability), 3),
        "risk_tier": risk_tier,
        "message": "🚨 High default risk — review carefully!" if prediction == 1
                   else "✅ Low default risk — likely to repay",
        "note": "Bureau and previous-application fields default to 'no history' unless explicitly provided"
    }