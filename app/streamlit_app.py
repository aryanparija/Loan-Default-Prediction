import streamlit as st
import pandas as pd
import joblib
import os

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Default Risk Scoring",
    page_icon="🏦",
    layout="wide"
)

# ── Load model ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(BASE_DIR, '../models/loan_default_model.pkl'))
    categorical_cols = joblib.load(os.path.join(BASE_DIR, '../models/categorical_cols.pkl'))
    numerical_cols = joblib.load(os.path.join(BASE_DIR, '../models/numerical_cols.pkl'))
    return model, categorical_cols, numerical_cols

model, categorical_cols, numerical_cols = load_model()

# ── Header ────────────────────────────────────────────────────
st.title("🏦 Loan Default Risk Scoring System")
st.markdown("Enter applicant details to predict default risk using LightGBM")
st.divider()

# ── Input form ───────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Personal Info")
    code_gender = st.selectbox("Gender", ["M", "F"])
    age_years = st.slider("Age", 18, 70, 35)
    cnt_children = st.number_input("Number of Children", 0, 10, 0)
    cnt_fam_members = st.number_input("Family Members", 1, 15, 2)
    name_family_status = st.selectbox("Family Status", 
        ["Married", "Single / not married", "Civil marriage", "Widow", "Separated"])
    name_education_type = st.selectbox("Education", 
        ["Secondary / secondary special", "Higher education", "Incomplete higher", 
         "Lower secondary", "Academic degree"])

with col2:
    st.subheader("Financial Info")
    amt_income_total = st.number_input("Annual Income (₹)", 25000, 5000000, 200000, step=10000)
    amt_credit = st.number_input("Loan Amount (₹)", 50000, 5000000, 400000, step=10000)
    amt_annuity = st.number_input("Monthly Annuity (₹)", 1000, 200000, 25000, step=1000)
    amt_goods_price = st.number_input("Goods Price (₹)", 50000, 5000000, 350000, step=10000)
    name_income_type = st.selectbox("Income Type", 
        ["Working", "Commercial associate", "Pensioner", "State servant", 
         "Unemployed", "Student", "Businessman", "Maternity leave"])
    flag_own_car = st.selectbox("Owns Car?", ["Y", "N"])

with col3:
    st.subheader("Employment & Credit Score")
    years_employed = st.slider("Years Employed", 0.0, 40.0, 5.0)
    ext_source_2 = st.slider("External Credit Score 2", 0.0, 1.0, 0.5)
    ext_source_3 = st.slider("External Credit Score 3", 0.0, 1.0, 0.5)
    occupation_type = st.selectbox("Occupation", 
        ["Laborers", "Sales staff", "Core staff", "Managers", "Drivers", 
         "High skill tech staff", "Accountants", "Medicine staff", "Other"])
    organization_type = st.selectbox("Organization Type",
        ["Business Entity Type 3", "Self-employed", "Other", "Government",
         "Trade: type 7", "School", "Medicine", "Industry: type 9"])

st.divider()

# ── Predict button ───────────────────────────────────────────
if st.button("🔍 Predict Default Risk", type="primary", use_container_width=True):
    
    # Build input dictionary with defaults for required fields
    input_dict = {
        'NAME_CONTRACT_TYPE': 'Cash loans',
        'CODE_GENDER': code_gender,
        'FLAG_OWN_CAR': flag_own_car,
        'FLAG_OWN_REALTY': 'Y',
        'CNT_CHILDREN': cnt_children,
        'AMT_INCOME_TOTAL': amt_income_total,
        'AMT_CREDIT': amt_credit,
        'AMT_ANNUITY': amt_annuity,
        'AMT_GOODS_PRICE': amt_goods_price,
        'NAME_TYPE_SUITE': 'Unaccompanied',
        'NAME_INCOME_TYPE': name_income_type,
        'NAME_EDUCATION_TYPE': name_education_type,
        'NAME_FAMILY_STATUS': name_family_status,
        'NAME_HOUSING_TYPE': 'House / apartment',
        'REGION_POPULATION_RELATIVE': 0.018,
        'DAYS_REGISTRATION': -3000.0,
        'OWN_CAR_AGE': 5.0 if flag_own_car == 'Y' else 0.0,
        'FLAG_MOBIL': 1,
        'FLAG_EMP_PHONE': 1,
        'FLAG_WORK_PHONE': 0,
        'FLAG_CONT_MOBILE': 1,
        'FLAG_PHONE': 1,
        'FLAG_EMAIL': 0,
        'OCCUPATION_TYPE': occupation_type,
        'CNT_FAM_MEMBERS': float(cnt_fam_members),
        'REGION_RATING_CLIENT': 2,
        'REGION_RATING_CLIENT_W_CITY': 2,
        'WEEKDAY_APPR_PROCESS_START': 'WEDNESDAY',
        'HOUR_APPR_PROCESS_START': 10,
        'REG_REGION_NOT_LIVE_REGION': 0,
        'REG_REGION_NOT_WORK_REGION': 0,
        'LIVE_REGION_NOT_WORK_REGION': 0,
        'REG_CITY_NOT_LIVE_CITY': 0,
        'REG_CITY_NOT_WORK_CITY': 0,
        'LIVE_CITY_NOT_WORK_CITY': 0,
        'ORGANIZATION_TYPE': organization_type,
        'EXT_SOURCE_2': ext_source_2,
        'EXT_SOURCE_3': ext_source_3,
        'DEF_30_CNT_SOCIAL_CIRCLE': 0.0,
        'DEF_60_CNT_SOCIAL_CIRCLE': 0.0,
        'DAYS_LAST_PHONE_CHANGE': -1000.0,
        'FLAG_DOCUMENT_3': 1,
        'FLAG_DOCUMENT_6': 0, 'FLAG_DOCUMENT_8': 0,
        'FLAG_DOCUMENT_2': 0, 'FLAG_DOCUMENT_4': 0, 'FLAG_DOCUMENT_5': 0,
        'FLAG_DOCUMENT_7': 0, 'FLAG_DOCUMENT_9': 0, 'FLAG_DOCUMENT_10': 0,
        'FLAG_DOCUMENT_11': 0, 'FLAG_DOCUMENT_12': 0, 'FLAG_DOCUMENT_13': 0,
        'FLAG_DOCUMENT_14': 0, 'FLAG_DOCUMENT_15': 0, 'FLAG_DOCUMENT_16': 0,
        'FLAG_DOCUMENT_17': 0, 'FLAG_DOCUMENT_18': 0, 'FLAG_DOCUMENT_19': 0,
        'FLAG_DOCUMENT_20': 0, 'FLAG_DOCUMENT_21': 0,
        'OBS_30_CNT_SOCIAL_CIRCLE': 0.0, 'OBS_60_CNT_SOCIAL_CIRCLE': 0.0,
        'DAYS_ID_PUBLISH': -1000.0,
        'AMT_REQ_CREDIT_BUREAU_HOUR': 0.0, 'AMT_REQ_CREDIT_BUREAU_DAY': 0.0,
        'AMT_REQ_CREDIT_BUREAU_WEEK': 0.0, 'AMT_REQ_CREDIT_BUREAU_MON': 0.0,
        'AMT_REQ_CREDIT_BUREAU_QRT': 0.0, 'AMT_REQ_CREDIT_BUREAU_YEAR': 0.0,
        'AGE_YEARS': float(age_years),
        'YEARS_EMPLOYED': float(years_employed),
        'CREDIT_INCOME_RATIO': amt_credit / (amt_income_total + 1),
        'ANNUITY_INCOME_RATIO': amt_annuity / (amt_income_total + 1),
        'CREDIT_GOODS_RATIO': amt_credit / (amt_goods_price + 1),
        'INCOME_PER_PERSON': amt_income_total / (cnt_fam_members + 1),
    }
    
    df = pd.DataFrame([input_dict])
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    default_probability = model.predict_proba(df)[0][1]
    prediction = model.predict(df)[0]
    
    if default_probability >= 0.75:
        risk_tier, color = "D — VERY HIGH RISK", "🔴"
    elif default_probability >= 0.50:
        risk_tier, color = "C — HIGH RISK", "🟠"
    elif default_probability >= 0.25:
        risk_tier, color = "B — MEDIUM RISK", "🟡"
    else:
        risk_tier, color = "A — LOW RISK", "🟢"
    
    st.divider()
    result_col1, result_col2, result_col3 = st.columns(3)
    
    with result_col1:
        st.metric("Default Probability", f"{default_probability:.1%}")
    with result_col2:
        st.metric("Risk Tier", f"{color} {risk_tier}")
    with result_col3:
        st.metric("Prediction", "Will Default" if prediction == 1 else "Will Repay")
    
    if prediction == 1:
        st.error("🚨 High default risk — review carefully before approval!")
    else:
        st.success("✅ Low default risk — likely to repay the loan")
    
    st.progress(float(default_probability))