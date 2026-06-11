# Loan Default Prediction

A production-grade machine learning system for predicting loan default risk, built on 307,511 real-world credit applications from Home Credit. The system outputs a calibrated default probability and risk tier (A/B/C/D) for each applicant, enabling data-driven credit decisions.

---

## Problem Statement

Credit default is one of the most significant sources of financial loss for banks and non-banking financial companies. Traditional rule-based credit scoring systems fail to capture complex non-linear relationships between applicant attributes and repayment behavior. This project addresses that gap by building an interpretable gradient boosting model that scores loan applications in real time via a REST API.

---

## Dataset

| Attribute | Details |
|-----------|---------|
| Source | Home Credit Default Risk — Kaggle |
| Size | 307,511 loan applications |
| Features | 122 (reduced to 75 after preprocessing) |
| Target | Binary — 1: defaulted, 0: repaid |
| Class distribution | 91.93% repaid / 8.07% defaulted |
| Data type | Real anonymized credit application data |

---

## Methodology

### 1. Exploratory Data Analysis
- Analyzed default rate across demographic and financial dimensions
- Key finding: Unemployed and maternity leave applicants exhibit 36–40% default rates — significantly above the 8% dataset average
- Identified 49 columns with >40% missing values, subsequently dropped
- Confirmed extreme class imbalance requiring specialized handling

### 2. Feature Engineering
Six domain-driven features were engineered from raw variables:

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `AGE_YEARS` | DAYS_BIRTH / -365 | Interpretable age in years |
| `YEARS_EMPLOYED` | DAYS_EMPLOYED / -365 | Employment stability |
| `CREDIT_INCOME_RATIO` | AMT_CREDIT / AMT_INCOME_TOTAL | Loan burden relative to income |
| `ANNUITY_INCOME_RATIO` | AMT_ANNUITY / AMT_INCOME_TOTAL | Monthly repayment burden |
| `CREDIT_GOODS_RATIO` | AMT_CREDIT / AMT_GOODS_PRICE | Loan-to-value ratio |
| `INCOME_PER_PERSON` | AMT_INCOME_TOTAL / CNT_FAM_MEMBERS | Effective income per dependent |

SHAP analysis post-training confirmed `CREDIT_GOODS_RATIO` as a top-3 predictor, validating the feature engineering approach.

### 3. Preprocessing Pipeline
Built using scikit-learn's `ColumnTransformer`:
- **Numerical:** `SimpleImputer` (median strategy) → `StandardScaler`
- **Categorical:** `SimpleImputer` (most frequent) → `OneHotEncoder` (handle_unknown='ignore')

### 4. Class Imbalance Strategy
Two approaches were evaluated:

| Approach | ROC-AUC | Notes |
|----------|---------|-------|
| SMOTE oversampling | 0.748 | Adds synthetic noise on large datasets |
| `scale_pos_weight=11` | 0.757 | Native LightGBM imbalance handling |

**Decision:** SMOTE was dropped in favor of `scale_pos_weight`. On datasets of 300K+ rows, gradient boosting's native class weighting consistently outperforms synthetic oversampling.

### 5. Model Comparison

| Model | ROC-AUC | Default Recall |
|-------|---------|---------------|
| Logistic Regression | 0.741 | 0.67 |
| Random Forest | 0.747 | 0.31 |
| XGBoost | 0.754 | 0.64 |
| **LightGBM (tuned)** | **0.757** | **0.67** |

### 6. Hyperparameter Tuning
GridSearchCV with 3-fold stratified cross-validation over 24 parameter combinations:

Best configuration:
learning_rate : 0.05
max_depth     : 8
n_estimators  : 200
num_leaves    : 31

### 7. Model Explainability
SHAP `TreeExplainer` was applied to interpret predictions at both global and local levels.

**Top predictors by mean absolute SHAP value:**
1. `EXT_SOURCE_3` — normalized external credit score
2. `EXT_SOURCE_2` — secondary external credit score
3. `CREDIT_GOODS_RATIO` — engineered loan-to-value feature
4. `YEARS_EMPLOYED` — employment tenure
5. `AGE_YEARS` — applicant age

---

## Results
Final Model   : LightGBM (GridSearchCV tuned)
ROC-AUC       : 0.757
Default Recall: 0.67
F1 Score      : 0.270

**Note on metrics:** Accuracy is not reported as it is not a meaningful metric for datasets with 92/8 class distributions. ROC-AUC and Recall are the primary evaluation metrics, consistent with industry practice for credit risk modeling.

---

## API Deployment

The trained pipeline is served via a FastAPI REST endpoint. The API accepts a JSON payload of applicant features and returns a default probability with a risk tier classification.

### Running Locally

```bash
pip install -r requirements.txt
cd app
uvicorn main:app --reload
```

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service status |
| `/health` | GET | Health check |
| `/predict` | POST | Score a loan application |
| `/docs` | GET | Interactive Swagger UI |

### Risk Tier Classification

| Tier | Default Probability | Recommended Action |
|------|--------------------|--------------------|
| A | < 0.25 | Approve |
| B | 0.25 – 0.50 | Review with standard checks |
| C | 0.50 – 0.75 | Enhanced due diligence required |
| D | > 0.75 | High likelihood of rejection |

### Sample Response

```json
{
  "default_predicted": true,
  "default_probability": 0.849,
  "risk_tier": "D — VERY HIGH RISK",
  "message": "🚨 High default risk — review carefully!"
}
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.12 |
| ML Framework | LightGBM, Scikit-learn |
| Imbalance Handling | scale_pos_weight (LightGBM native) |
| Explainability | SHAP TreeExplainer |
| API Framework | FastAPI |
| Server | Uvicorn |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Model Serialization | Joblib |

---

## Project Structure
loan-default-prediction/
├── data/
│   ├── application_train.csv
│   └── HomeCredit_columns_description.csv
├── notebooks/
│   └── 01_eda.ipynb
├── models/
│   ├── loan_default_model.pkl
│   ├── categorical_cols.pkl
│   └── numerical_cols.pkl
├── app/
│   └── main.py
├── requirements.txt
└── README.md

---

## Author

**Aryan Parija**
[LinkedIn](https://www.linkedin.com/in/aryanparija2006/) | [GitHub](https://github.com/aryanparija)