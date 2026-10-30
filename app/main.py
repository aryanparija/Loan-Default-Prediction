from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import joblib
import pandas as pd
import numpy as np
import os

# --- Load model -------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "../models/loan_default_model.pkl"))
categorical_cols = joblib.load(os.path.join(BASE_DIR, "../models/categorical_cols.pkl"))
numerical_cols = joblib.load(os.path.join(BASE_DIR, "../models/numerical_cols.pkl"))

app = FastAPI(
    title="Loan Default Risk Scoring & Decision-Theoretic API",
    description=(
        "Predicts loan default risk using LightGBM enriched with multi-relational credit history. "
        "Returns both cost-weighted risk scores, calibrated posterior default probabilities (PD), "
        "and decision-theoretic approval recommendations across 5:1, 10:1, and 15:1 cost ratios."
    ),
    version="3.0.0",
)


def calibrate_probability(raw_weighted_prob: float, pos_weight: float = 11.39) -> float:
    p = float(np.clip(raw_weighted_prob, 1e-6, 1.0 - 1e-6))
    odds_weighted = p / (1.0 - p)
    odds_true = odds_weighted / pos_weight
    return float(odds_true / (1.0 + odds_true))


class LoanInput(BaseModel):
    NAME_CONTRACT_TYPE: str = "Cash loans"
    CODE_GENDER: str
    FLAG_OWN_CAR: str
    FLAG_OWN_REALTY: str = "Y"
    CNT_CHILDREN: int = 0
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: float
    AMT_GOODS_PRICE: float
    NAME_TYPE_SUITE: str = "Unaccompanied"
    NAME_INCOME_TYPE: str
    NAME_EDUCATION_TYPE: str
    NAME_FAMILY_STATUS: str
    NAME_HOUSING_TYPE: str = "House / apartment"
    REGION_POPULATION_RELATIVE: float = 0.018
    DAYS_REGISTRATION: float = -3000.0
    OWN_CAR_AGE: float = 0.0
    FLAG_MOBIL: int = 1
    FLAG_EMP_PHONE: int = 1
    FLAG_WORK_PHONE: int = 0
    FLAG_CONT_MOBILE: int = 1
    FLAG_PHONE: int = 1
    FLAG_EMAIL: int = 0
    OCCUPATION_TYPE: str = "Laborers"
    CNT_FAM_MEMBERS: float = 2.0
    REGION_RATING_CLIENT: int = 2
    REGION_RATING_CLIENT_W_CITY: int = 2
    WEEKDAY_APPR_PROCESS_START: str = "WEDNESDAY"
    HOUR_APPR_PROCESS_START: int = 10
    REG_REGION_NOT_LIVE_REGION: int = 0
    REG_REGION_NOT_WORK_REGION: int = 0
    LIVE_REGION_NOT_WORK_REGION: int = 0
    REG_CITY_NOT_LIVE_CITY: int = 0
    REG_CITY_NOT_WORK_CITY: int = 0
    LIVE_CITY_NOT_WORK_CITY: int = 0
    ORGANIZATION_TYPE: str = "Business Entity Type 3"
    EXT_SOURCE_1: Optional[float] = None
    EXT_SOURCE_2: float
    EXT_SOURCE_3: float
    DEF_30_CNT_SOCIAL_CIRCLE: float = 0.0
    DEF_60_CNT_SOCIAL_CIRCLE: float = 0.0
    DAYS_LAST_PHONE_CHANGE: float = -1000.0
    FLAG_DOCUMENT_3: int = 1
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

    # --- Bureau history features ---
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

    # --- Previous Home Credit application features ---
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

    # --- Optional Installment / POS / Credit Card behavioral features ---
    INSTAL_LATE_RATE: float = 0.0
    CC_UTILIZATION_MEAN: float = 0.0
    COST_RATIO_FN_FP: int = 10


@app.get("/")
def home():
    return {
        "message": "Loan Default Risk Scoring API v3.0.0 is running!",
        "benchmark_roc_auc": 0.7876,
        "calibrated_ece": 0.0018,
        "endpoints": {"predict": "/predict", "health": "/health", "docs": "/docs"},
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "LightGBM Multi-Relational Loan Default Predictor (7-Table Benchmark ROC-AUC 0.7876)",
    }


@app.post("/predict")
def predict(loan: LoanInput):
    input_dict = loan.model_dump() if hasattr(loan, "model_dump") else loan.dict()
    cost_ratio = input_dict.pop("COST_RATIO_FN_FP", 10)

    ext_vals = [v for v in [input_dict.get("EXT_SOURCE_1"), input_dict.get("EXT_SOURCE_2"), input_dict.get("EXT_SOURCE_3")] if v is not None]
    input_dict["EXT_SOURCE_MEAN"] = float(np.mean(ext_vals)) if ext_vals else 0.5
    input_dict["EXT_SOURCE_STD"] = float(np.std(ext_vals)) if len(ext_vals) > 1 else 0.0
    input_dict["CREDIT_TO_ANNUITY_RATIO"] = input_dict["AMT_CREDIT"] / (input_dict["AMT_ANNUITY"] + 1.0)
    input_dict["EMPLOYED_TO_AGE_RATIO"] = input_dict["YEARS_EMPLOYED"] / (input_dict["AGE_YEARS"] + 0.01)

    df = pd.DataFrame([input_dict])
    df["NO_BUREAU_HISTORY"] = (df["BUREAU_LOAN_COUNT"] == 0).astype(int)
    df["NO_PREV_APPLICATION"] = (df["PREV_APP_COUNT"] == 0).astype(int)

    for col in numerical_cols:
        if col not in df.columns:
            df[col] = 0.0 if col.startswith(("FLAG_", "NO_", "BUREAU_", "PREV_", "INSTAL_", "POS_", "CC_")) else np.nan
    for col in categorical_cols:
        if col not in df.columns:
            df[col] = "missing"
        df[col] = df[col].astype(str)

    raw_weighted_prob = float(model.predict_proba(df)[0][1])
    calibrated_pd = calibrate_probability(raw_weighted_prob)

    if cost_ratio <= 5:
        optimal_t = 0.175
    elif cost_ratio >= 15:
        optimal_t = 0.065
    else:
        optimal_t = 0.080

    default_predicted = bool(calibrated_pd >= optimal_t)

    if raw_weighted_prob >= 0.75:
        risk_tier = "D - VERY HIGH RISK"
    elif raw_weighted_prob >= 0.50:
        risk_tier = "C - HIGH RISK"
    elif raw_weighted_prob >= 0.25:
        risk_tier = "B - MEDIUM RISK"
    else:
        risk_tier = "A - LOW RISK"

    return {
        "default_predicted": default_predicted,
        "default_probability": round(raw_weighted_prob, 4),
        "calibrated_pd": round(calibrated_pd, 4),
        "optimal_threshold_t": optimal_t,
        "cost_ratio_fn_fp": f"{cost_ratio}:1",
        "risk_tier": risk_tier,
        "message": (
            "🚨 High default risk – route to manual underwriting review!"
            if default_predicted
            else "✅ Low default risk – recommended for approval"
        ),
    }