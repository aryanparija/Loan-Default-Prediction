import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --- Page config -------------------------------------------
st.set_page_config(
    page_title="Loan Default Risk Scoring & Decision Support",
    page_icon="🏦",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(BASE_DIR, "../models/loan_default_model.pkl"))
    categorical_cols = joblib.load(os.path.join(BASE_DIR, "../models/categorical_cols.pkl"))
    numerical_cols = joblib.load(os.path.join(BASE_DIR, "../models/numerical_cols.pkl"))
    return model, categorical_cols, numerical_cols


model, categorical_cols, numerical_cols = load_model()


def calibrate_probability(raw_weighted_prob: float, pos_weight: float = 11.39) -> float:
    """
    Convert cost-weighted model probability (trained with scale_pos_weight ~ 11.4)
    to prior-corrected posterior probability of default (PD), matching Platt/Bayes scaling.
    """
    p = float(np.clip(raw_weighted_prob, 1e-6, 1.0 - 1e-6))
    odds_weighted = p / (1.0 - p)
    odds_true = odds_weighted / pos_weight
    return float(odds_true / (1.0 + odds_true))


# --- Header -----------------------------------------------
st.title("🏦 Loan Default Risk Scoring & Decision-Theoretic Underwriting System")
st.markdown(
    "**Powered by Enriched LightGBM (7-Table Benchmark ROC-AUC: `0.7876` | Calibrated ECE: `0.0018`)** — "
    "Enter applicant demographics, financial capacity, external bureau history, and decision-cost parameters below."
)
st.divider()

# --- Main Input Columns -----------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("1. Personal & Household Info")
    code_gender = st.selectbox("Gender", ["M", "F"])
    age_years = st.slider("Applicant Age (Years)", 18, 70, 35)
    cnt_children = st.number_input("Number of Children", 0, 10, 0)
    cnt_fam_members = st.number_input("Total Family Members", 1, 15, 2)
    name_family_status = st.selectbox(
        "Family Status",
        ["Married", "Single / not married", "Civil marriage", "Widow", "Separated"],
    )
    name_education_type = st.selectbox(
        "Education Level",
        [
            "Secondary / secondary special",
            "Higher education",
            "Incomplete higher",
            "Lower secondary",
            "Academic degree",
        ],
    )

with col2:
    st.subheader("2. Loan & Financial Capacity")
    amt_income_total = st.number_input("Annual Income (₹)", 25000, 10000000, 200000, step=10000)
    amt_credit = st.number_input("Requested Loan Credit (₹)", 50000, 10000000, 400000, step=10000)
    amt_annuity = st.number_input("Annualized Loan Annuity (₹)", 1000, 500000, 25000, step=1000)
    amt_goods_price = st.number_input("Financed Goods Price (₹)", 50000, 10000000, 350000, step=10000)
    name_income_type = st.selectbox(
        "Income Type",
        [
            "Working",
            "Commercial associate",
            "Pensioner",
            "State servant",
            "Unemployed",
            "Student",
            "Businessman",
            "Maternity leave",
        ],
    )
    flag_own_car = st.selectbox("Owns a Car?", ["Y", "N"])

with col3:
    st.subheader("3. Employment & External Scores")
    years_employed = st.slider("Years Employed", 0.0, 45.0, 5.0, step=0.5)
    ext_source_1 = st.slider("External Bureau Score 1 (EXT_SOURCE_1)", 0.0, 1.0, 0.50, step=0.01)
    ext_source_2 = st.slider("External Bureau Score 2 (EXT_SOURCE_2)", 0.0, 1.0, 0.50, step=0.01)
    ext_source_3 = st.slider("External Bureau Score 3 (EXT_SOURCE_3)", 0.0, 1.0, 0.50, step=0.01)
    occupation_type = st.selectbox(
        "Occupation Type",
        [
            "Laborers",
            "Sales staff",
            "Core staff",
            "Managers",
            "Drivers",
            "High skill tech staff",
            "Accountants",
            "Medicine staff",
            "Other",
        ],
    )
    organization_type = st.selectbox(
        "Employer Organization Type",
        [
            "Business Entity Type 3",
            "Self-employed",
            "Other",
            "Government",
            "Trade: type 7",
            "School",
            "Medicine",
            "Industry: type 9",
        ],
    )

# --- Optional Expandable Relational Credit History & Cost Policy ---
with st.expander("📊 4. Auxiliary Credit Bureau, Prior Loan & Installment History (Optional — Defaults to Thin-File)", expanded=False):
    bcol1, bcol2, bcol3 = st.columns(3)
    with bcol1:
        st.markdown("**External Credit Bureau (`bureau.csv`)**")
        bureau_loan_count = st.number_input("Prior Bureau Loans Count", 0, 50, 0)
        bureau_active_loans = st.number_input("Active Bureau Loans", 0, 20, 0)
        bureau_total_credit_sum = st.number_input("Total Bureau Credit Sum (₹)", 0.0, 50000000.0, 0.0, step=50000.0)
        bureau_total_debt = st.number_input("Total Bureau Outstanding Debt (₹)", 0.0, 50000000.0, 0.0, step=25000.0)
        bureau_max_overdue_days = st.number_input("Max Historical Overdue Days (DPD)", 0, 1000, 0)
    with bcol2:
        st.markdown("**Prior Home Credit Loans (`previous_application.csv`)**")
        prev_app_count = st.number_input("Prior Home Credit Applications", 0, 30, 0)
        prev_approved_count = st.number_input("Prior Approved Applications", 0, 30, 0)
        prev_refused_count = st.number_input("Prior Refused Applications", 0, 30, 0)
        prev_cancelled_count = st.number_input("Prior Cancelled Applications", 0, 30, 0)
    with bcol3:
        st.markdown("**Repayment & Decision Cost Policy**")
        instal_late_rate = st.slider("Historical Installment Late Rate", 0.0, 1.0, 0.0, step=0.05)
        cc_utilization_mean = st.slider("Mean Credit Card Limit Utilization", 0.0, 1.2, 0.0, step=0.05)
        cost_policy = st.selectbox(
            "Decision-Theoretic Cost Structure (FN : FP Ratio)",
            [
                "Standard 10:1 Cost Ratio (Calibrated Cutoff t* = 0.080 | Weighted t = 0.50)",
                "Moderate 5:1 Cost Ratio (Calibrated Cutoff t* = 0.175 | Max F1 = 0.338)",
                "Conservative 15:1 Cost Ratio (Calibrated Cutoff t* = 0.065 | Recall = 79.1%)",
            ],
        )

st.divider()

if st.button("🔍 Score Loan Application", type="primary", use_container_width=True):
    no_bureau = 1 if bureau_loan_count == 0 else 0
    no_prev = 1 if prev_app_count == 0 else 0
    bureau_debt_credit_ratio = bureau_total_debt / (bureau_total_credit_sum + 1.0) if bureau_loan_count > 0 else 0.0
    prev_refusal_rate = prev_refused_count / (prev_app_count + 1.0) if prev_app_count > 0 else 0.0

    input_dict = {
        "NAME_CONTRACT_TYPE": "Cash loans",
        "CODE_GENDER": code_gender,
        "FLAG_OWN_CAR": flag_own_car,
        "FLAG_OWN_REALTY": "Y",
        "CNT_CHILDREN": cnt_children,
        "AMT_INCOME_TOTAL": float(amt_income_total),
        "AMT_CREDIT": float(amt_credit),
        "AMT_ANNUITY": float(amt_annuity),
        "AMT_GOODS_PRICE": float(amt_goods_price),
        "NAME_TYPE_SUITE": "Unaccompanied",
        "NAME_INCOME_TYPE": name_income_type,
        "NAME_EDUCATION_TYPE": name_education_type,
        "NAME_FAMILY_STATUS": name_family_status,
        "NAME_HOUSING_TYPE": "House / apartment",
        "REGION_POPULATION_RELATIVE": 0.018,
        "DAYS_REGISTRATION": -3000.0,
        "OWN_CAR_AGE": 5.0 if flag_own_car == "Y" else 0.0,
        "FLAG_MOBIL": 1,
        "FLAG_EMP_PHONE": 1,
        "FLAG_WORK_PHONE": 0,
        "FLAG_CONT_MOBILE": 1,
        "FLAG_PHONE": 1,
        "FLAG_EMAIL": 0,
        "OCCUPATION_TYPE": occupation_type,
        "CNT_FAM_MEMBERS": float(cnt_fam_members),
        "REGION_RATING_CLIENT": 2,
        "REGION_RATING_CLIENT_W_CITY": 2,
        "WEEKDAY_APPR_PROCESS_START": "WEDNESDAY",
        "HOUR_APPR_PROCESS_START": 10,
        "REG_REGION_NOT_LIVE_REGION": 0,
        "REG_REGION_NOT_WORK_REGION": 0,
        "LIVE_REGION_NOT_WORK_REGION": 0,
        "REG_CITY_NOT_LIVE_CITY": 0,
        "REG_CITY_NOT_WORK_CITY": 0,
        "LIVE_CITY_NOT_WORK_CITY": 0,
        "ORGANIZATION_TYPE": organization_type,
        "EXT_SOURCE_1": float(ext_source_1),
        "EXT_SOURCE_2": float(ext_source_2),
        "EXT_SOURCE_3": float(ext_source_3),
        "EXT_SOURCE_MEAN": float(np.mean([ext_source_1, ext_source_2, ext_source_3])),
        "EXT_SOURCE_STD": float(np.std([ext_source_1, ext_source_2, ext_source_3])),
        "DEF_30_CNT_SOCIAL_CIRCLE": 0.0,
        "DEF_60_CNT_SOCIAL_CIRCLE": 0.0,
        "DAYS_LAST_PHONE_CHANGE": -1000.0,
        "FLAG_DOCUMENT_3": 1,
        "OBS_30_CNT_SOCIAL_CIRCLE": 0.0,
        "OBS_60_CNT_SOCIAL_CIRCLE": 0.0,
        "DAYS_ID_PUBLISH": -1000.0,
        "AMT_REQ_CREDIT_BUREAU_HOUR": 0.0,
        "AMT_REQ_CREDIT_BUREAU_DAY": 0.0,
        "AMT_REQ_CREDIT_BUREAU_WEEK": 0.0,
        "AMT_REQ_CREDIT_BUREAU_MON": 0.0,
        "AMT_REQ_CREDIT_BUREAU_QRT": 0.0,
        "AMT_REQ_CREDIT_BUREAU_YEAR": 0.0,
        "AGE_YEARS": float(age_years),
        "YEARS_EMPLOYED": float(years_employed),
        "DAYS_EMPLOYED_ANOM": 0,
        "CREDIT_INCOME_RATIO": amt_credit / (amt_income_total + 1.0),
        "ANNUITY_INCOME_RATIO": amt_annuity / (amt_income_total + 1.0),
        "CREDIT_GOODS_RATIO": amt_credit / (amt_goods_price + 1.0),
        "INCOME_PER_PERSON": amt_income_total / (cnt_fam_members + 1.0),
        "CREDIT_TO_ANNUITY_RATIO": amt_credit / (amt_annuity + 1.0),
        "EMPLOYED_TO_AGE_RATIO": years_employed / (age_years + 0.01),
        "BUREAU_LOAN_COUNT": float(bureau_loan_count),
        "BUREAU_ACTIVE_LOANS": float(bureau_active_loans),
        "BUREAU_CREDIT_TYPES": float(min(bureau_loan_count, 2)),
        "BUREAU_DAYS_CREDIT_MEAN": -700.0 if bureau_loan_count > 0 else np.nan,
        "BUREAU_DAYS_CREDIT_MIN": -1200.0 if bureau_loan_count > 0 else np.nan,
        "BUREAU_MAX_OVERDUE_DAYS": float(bureau_max_overdue_days),
        "BUREAU_MAX_AMT_OVERDUE": 0.0,
        "BUREAU_TOTAL_PROLONG": 0.0,
        "BUREAU_TOTAL_CREDIT_SUM": float(bureau_total_credit_sum),
        "BUREAU_TOTAL_DEBT": float(bureau_total_debt),
        "BUREAU_TOTAL_OVERDUE": 0.0,
        "BUREAU_DEBT_CREDIT_RATIO": float(bureau_debt_credit_ratio),
        "NO_BUREAU_HISTORY": int(no_bureau),
        "PREV_APP_COUNT": float(prev_app_count),
        "PREV_APPROVED_COUNT": float(prev_approved_count),
        "PREV_REFUSED_COUNT": float(prev_refused_count),
        "PREV_CANCELLED_COUNT": float(prev_cancelled_count),
        "PREV_DAYS_DECISION_MEAN": -500.0 if prev_app_count > 0 else np.nan,
        "PREV_DAYS_DECISION_MIN": -900.0 if prev_app_count > 0 else np.nan,
        "PREV_AMT_APPLICATION_MEAN": float(amt_credit) if prev_app_count > 0 else np.nan,
        "PREV_AMT_CREDIT_MEAN": float(amt_credit) if prev_app_count > 0 else np.nan,
        "PREV_AMT_ANNUITY_MEAN": float(amt_annuity) if prev_app_count > 0 else np.nan,
        "PREV_CNT_PAYMENT_MEAN": 12.0 if prev_app_count > 0 else np.nan,
        "PREV_RATE_DOWN_PAYMENT_MEAN": 0.05 if prev_app_count > 0 else np.nan,
        "PREV_CREDIT_APPLICATION_RATIO": 1.0 if prev_app_count > 0 else np.nan,
        "PREV_REFUSAL_RATE": float(prev_refusal_rate),
        "NO_PREV_APPLICATION": int(no_prev),
        "INSTAL_LATE_RATE": float(instal_late_rate),
        "CC_UTILIZATION_MEAN": float(cc_utilization_mean),
    }

    # Ensure every column expected by the serialized pipeline is populated
    for col in numerical_cols:
        if col not in input_dict:
            input_dict[col] = 0.0 if col.startswith(("FLAG_", "NO_", "BUREAU_", "PREV_", "INSTAL_", "POS_", "CC_")) else np.nan
    for col in categorical_cols:
        if col not in input_dict:
            input_dict[col] = "missing"

    df_in = pd.DataFrame([input_dict])
    for col in categorical_cols:
        if col in df_in.columns:
            df_in[col] = df_in[col].astype(str)

    raw_weighted_prob = float(model.predict_proba(df_in)[0][1])
    calibrated_pd = calibrate_probability(raw_weighted_prob)

    if "5:1" in cost_policy:
        cal_thresh = 0.175
    elif "15:1" in cost_policy:
        cal_thresh = 0.065
    else:
        cal_thresh = 0.080

    is_default = calibrated_pd >= cal_thresh

    if raw_weighted_prob >= 0.75:
        risk_tier, color = "D – VERY HIGH RISK", "🔴"
    elif raw_weighted_prob >= 0.50:
        risk_tier, color = "C – HIGH RISK", "🟠"
    elif raw_weighted_prob >= 0.25:
        risk_tier, color = "B – MEDIUM RISK", "🟡"
    else:
        risk_tier, color = "A – LOW RISK", "🟢"

    st.divider()
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.metric("Weighted Risk Score (w+=11)", f"{raw_weighted_prob:.1%}")
    with r2:
        st.metric("Calibrated Empirical PD", f"{calibrated_pd:.2%}", help="Prior-calibrated probability of default (true portfolio base rate = 8.07%)")
    with r3:
        st.metric("Underwriting Risk Tier", f"{color} {risk_tier}")
    with r4:
        st.metric(f"Decision (t* = {cal_thresh:.3f})", "Flag for Review / Decline" if is_default else "Approve Loan")

    if is_default:
        st.error(
            f"🚨 **Elevated Default Risk:** Calibrated PD ({calibrated_pd:.2%}) exceeds the optimal decision-theoretic threshold "
            f"(`t* = {cal_thresh:.3f}`). Route to manual underwriting or request collateral."
        )
    else:
        st.success(
            f"✅ **Creditworthy Profile:** Calibrated PD ({calibrated_pd:.2%}) is below the decision-theoretic cutoff "
            f"(`t* = {cal_thresh:.3f}`). Recommended for approval."
        )