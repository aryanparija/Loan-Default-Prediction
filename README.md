# Loan Default Prediction

A production-grade machine learning system for predicting loan default risk, built on 307,511 real-world credit applications from Home Credit, enriched with applicant credit bureau history and prior loan application records. The system outputs a calibrated default probability and risk tier (A/B/C/D) for each applicant, enabling data-driven credit decisions.

---

**🔗 Live Demo:** https://loan-default-prediction-aryanparija.streamlit.app/

## Problem Statement

Credit default is one of the most significant sources of financial loss for banks and non-banking financial companies. Traditional rule-based credit scoring systems fail to capture complex non-linear relationships between applicant attributes and repayment behavior. This project addresses that gap by building an interpretable gradient boosting model that scores loan applications in real time via a REST API and a Streamlit interface.

---

## Dataset

| Attribute | Details |
|-----------|---------|
| Source | Home Credit Default Risk — Kaggle |
| Primary table | `application_train.csv` — 307,511 loan applications, 122 raw features |
| Auxiliary table 1 | `bureau.csv` — 1,716,428 records, 305,811 unique applicants (external credit bureau history) |
| Auxiliary table 2 | `previous_application.csv` — 1,670,214 records, 338,857 unique applicants (prior Home Credit applications) |
| Final feature count | 102 (after merging auxiliary tables and dropping high-missing columns) |
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
Six domain-driven features were engineered from raw application-level variables:

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `AGE_YEARS` | DAYS_BIRTH / -365 | Interpretable age in years |
| `YEARS_EMPLOYED` | DAYS_EMPLOYED / -365 | Employment stability |
| `CREDIT_INCOME_RATIO` | AMT_CREDIT / AMT_INCOME_TOTAL | Loan burden relative to income |
| `ANNUITY_INCOME_RATIO` | AMT_ANNUITY / AMT_INCOME_TOTAL | Monthly repayment burden |
| `CREDIT_GOODS_RATIO` | AMT_CREDIT / AMT_GOODS_PRICE | Loan-to-value ratio |
| `INCOME_PER_PERSON` | AMT_INCOME_TOTAL / CNT_FAM_MEMBERS | Effective income per dependent |

### 3. Auxiliary Data Enrichment

The baseline model used only `application_train.csv`. Two additional Home Credit tables were incorporated to capture credit history the base table doesn't contain:

**`bureau.csv`** — the applicant's credit history with *other* financial institutions, reported to the credit bureau. One-to-many relative to the main table (average 5.6 records per applicant). Aggregated to one row per applicant via `groupby('SK_ID_CURR')`: loan counts, active loan counts, total debt/credit exposure, days since credit opened, overdue history, and a debt-to-credit ratio.

**`previous_application.csv`** — the applicant's own prior loan applications with Home Credit itself. Also one-to-many (average 4.9 records per applicant). Aggregated to: approval/refusal/cancellation counts, historical refusal rate, mean requested vs. approved credit amounts, and days since the last decision.

**Aggregation and missing-value strategy** — handled column-by-column rather than with a blanket rule:
- Counts and sums (e.g. `BUREAU_LOAN_COUNT`, `PREV_REFUSED_COUNT`) were zero-filled after merging, since a missing aggregate genuinely means zero occurrences for that applicant
- Mean/min-based columns (e.g. `BUREAU_DAYS_CREDIT_MEAN`, `PREV_DAYS_DECISION_MEAN`) were left as `NaN` and handled by the pipeline's existing `SimpleImputer(median)`, since zero-filling a "days" column would falsely sit inside the real distribution rather than at a meaningful extreme
- Two explicit indicator flags, `NO_BUREAU_HISTORY` and `NO_PREV_APPLICATION`, were added to distinguish "no history at all" from any numeric fill — this is itself a real, informative signal rather than an artifact of imputation

**Key finding — the two "no history" flags behave in opposite directions**, consistent with known credit risk patterns:
- `NO_BUREAU_HISTORY` correlates **positively** with default risk — applicants with no external credit history ("thin-file") are statistically harder to assess and riskier
- `NO_PREV_APPLICATION` correlates **negatively** with default risk — being new to *this specific lender* doesn't carry the same penalty, since a well-established borrower elsewhere may simply never have applied to Home Credit before

This distinction was validated visually in `01_eda.ipynb` (refusal-rate default staircase, thin-file vs. first-time-applicant comparison) before being carried into modeling.

### 4. Preprocessing Pipeline
Built using scikit-learn's `ColumnTransformer`:
- **Numerical:** `SimpleImputer` (median strategy) → `StandardScaler`
- **Categorical:** `SimpleImputer` (most frequent) → `OneHotEncoder` (handle_unknown='ignore')

### 5. Class Imbalance Strategy
Two approaches were evaluated on the baseline feature set:

| Approach | ROC-AUC | Notes |
|----------|---------|-------|
| SMOTE oversampling | 0.748 | Adds synthetic noise on large datasets |
| `scale_pos_weight=11` | 0.757 | Native LightGBM imbalance handling |

**Decision:** SMOTE was dropped in favor of `scale_pos_weight`. On datasets of 300K+ rows, gradient boosting's native class weighting consistently outperforms synthetic oversampling. This setting was retained for the enriched model.

### 6. Model Comparison

Models were first compared on the enriched 102-feature set without tuning, to confirm the auxiliary tables added value before investing in hyperparameter search:

| Model | ROC-AUC (baseline, 75 features) | ROC-AUC (enriched, 102 features) | Change |
|-------|-------------------------------|-----------------------------------|--------|
| Logistic Regression | 0.741 | 0.757 | +0.016 |
| Random Forest | 0.747 | 0.757 | +0.010 |
| XGBoost | 0.754 | 0.764 | +0.010 |
| **LightGBM** | **0.757** | **0.764** | +0.007 |

The improvement was consistent across four structurally different model types (linear, bagged trees, two boosting implementations), which is stronger evidence of a genuine signal than a gain on a single architecture would be.

### 7. Hyperparameter Tuning

GridSearchCV with 3-fold cross-validation was run on LightGBM using the enriched feature set:

```
Best configuration:
  learning_rate : 0.05
  max_depth     : 8
  n_estimators  : 200
  num_leaves    : 31
```

### 8. Model Explainability — What Actually Mattered

SHAP `TreeExplainer` was applied to interpret the tuned model's predictions at both global and local levels.

6 of the top 20 features by mean absolute SHAP value are derived from the auxiliary tables: `PREV_CNT_PAYMENT_MEAN`, `PREV_CREDIT_APPLICATION_RATIO`, `BUREAU_DEBT_CREDIT_RATIO`, `BUREAU_TOTAL_CREDIT_SUM`, `PREV_AMT_ANNUITY_MEAN`, and `PREV_REFUSAL_RATE`.

Notably, `BUREAU_DEBT_CREDIT_RATIO` and `BUREAU_TOTAL_CREDIT_SUM` showed near-zero linear correlation with the target during EDA (under 0.002) but rank in the top 10–13 by SHAP importance — confirming that LightGBM captured non-linear relationships in these features that a simple correlation check could not detect. This is the reason feature selection was deferred to post-training SHAP analysis rather than pre-filtering on correlation strength alone.

`EXT_SOURCE_2` and `EXT_SOURCE_3` (external bureau-derived credit scores already present in the base application table) remain the dominant predictors by a wide margin — the auxiliary table features supplement rather than replace this existing signal.

---

## Results

```
Final Model     : LightGBM (GridSearchCV tuned, enriched feature set)
Features        : 102 (application + bureau + previous_application)
ROC-AUC         : 0.770
Default Recall  : 0.67
F1 Score        : 0.279
```

| Stage | Features | ROC-AUC | F1 | Default Recall |
|-------|----------|---------|-----|-----------------|
| Baseline (application data only) | 75 | 0.757 | 0.270 | 0.67 |
| **Enriched (+ bureau, + previous_application, tuned)** | **102** | **0.770** | **0.279** | **0.67** |

**Note on metrics:** Accuracy is not reported as it is not a meaningful metric for datasets with 92/8 class distributions. ROC-AUC and Recall are the primary evaluation metrics, consistent with industry practice for credit risk modeling. Recall was deliberately held constant at 0.67 across both stages — the enrichment improved how risk is *ranked* (ROC-AUC), without trading away the recall-first threshold decision described below.

---

## Why F1 is 0.279 — The Precision-Recall Tradeoff

F1 of 0.279 reflects a deliberate business tradeoff. At the default threshold (0.5), the model achieves 67% recall with roughly 18% precision. In credit risk modeling, missing a real defaulter is significantly more costly than a false rejection — a bad loan costs the bank the entire principal, while a false rejection costs only one lost customer. Recall is therefore the primary optimization target. This threshold choice is consistent with standard industry practice in credit scoring.

---

## Why ROC-AUC is 0.770, Not Higher — Benchmarked Against the Leaderboard

This project incorporates 3 of 7 available Home Credit tables: `application_train.csv`, `bureau.csv`, and `previous_application.csv`. The four tables deliberately excluded — `bureau_balance.csv`, `POS_CASH_balance.csv`, `credit_card_balance.csv`, and `installments_payments.csv` — are time-series tables requiring substantially more complex feature engineering (or recurrent neural networks, as used in several top public solutions) to extract value from.

Public Kaggle solutions for this competition achieve ROC-AUC up to ~0.80 on the private leaderboard (top 0.2%, solo) by incorporating all 7 tables, training GRU networks on the time-series tables, applying genetic feature synthesis, and ensembling multiple models — work spanning months, not a single project iteration.

This project scoped to the two highest-value auxiliary tables to balance signal value against engineering complexity — a deliberate, documented tradeoff rather than an oversight. The resulting **+0.013 ROC-AUC improvement (0.757 → 0.770)** is methodologically sound and consistent in direction across all four models tested, but represents roughly 30% of the ~0.045 gap between this baseline and a top-tier leaderboard solution. Closing the remaining gap is the clearest path forward and is listed under Known Limitations below.

---

## How to Reproduce

1. Download the dataset from Kaggle:
   https://www.kaggle.com/competitions/home-credit-default-risk/data
   Place `application_train.csv`, `bureau.csv`, and `previous_application.csv` in the `data/` folder

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run notebooks in order:
   - `notebooks/01_eda.ipynb` — exploratory data analysis, including bureau/previous_application feature EDA
   - `notebooks/02_modeling.ipynb` — auxiliary table aggregation, preprocessing, training, GridSearchCV, SHAP

4. Start the API:
   ```bash
   cd app
   uvicorn main:app --reload
   ```

5. Open Swagger UI at `http://127.0.0.1:8000/docs`

6. Or run the Streamlit interface:
   ```bash
   cd app
   streamlit run streamlit_app.py
   ```

**Note:** `application_train.csv`, `bureau.csv`, and `previous_application.csv` are not included in this repository due to GitHub file size limits (300MB+ combined). Download directly from Kaggle using the link above.

---

## Known Limitations and Future Improvements

| Limitation | Explanation | Fix |
|------------|-------------|-----|
| ROC-AUC 0.770 vs. ~0.80 leaderboard ceiling | Only 3 of 7 Home Credit tables used; 4 time-series tables excluded | Incorporate `bureau_balance.csv`, `POS_CASH_balance.csv`, `credit_card_balance.csv`, `installments_payments.csv`, likely via sequence models or further aggregation |
| F1 0.279 | Precision-recall tradeoff at 0.5 threshold | Tune threshold per an explicit business cost matrix |
| Probability calibration | `scale_pos_weight` can produce uncalibrated probabilities | Apply `CalibratedClassifierCV` |
| GridSearchCV scope | 16 combinations, 3-fold CV — lightweight tuning | Use Optuna or a Bayesian search for a wider, more efficient search |
| API/UI demo inputs | The deployed API and Streamlit app default all bureau/previous-application fields to "no history" rather than collecting them from the user, since this aggregate data would normally be looked up server-side from internal records rather than typed in by an applicant | Document explicitly (see note below); not a defect, a deliberate UX scoping decision |

**Note on deployment inputs:** the live demo and API expose only the original application-level fields as user inputs. The 27 bureau/previous-application fields the enriched model expects are auto-defaulted to "no history" values (`NO_BUREAU_HISTORY=1`, `NO_PREV_APPLICATION=1`, with counts/sums at 0 and true-missing fields imputed by the pipeline). This mirrors how a real credit system would behave — internal bureau and application history is looked up automatically by applicant ID, never typed in manually — but it does mean demo predictions reflect a "first-time, no external history" applicant profile rather than incorporating real bureau data, even when the underlying model was trained to use it.

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
  "default_probability": 0.826,
  "risk_tier": "D — VERY HIGH RISK",
  "message": "🚨 High default risk — review carefully!",
  "note": "Bureau and previous-application fields default to 'no history' unless explicitly provided"
}
```

---

## Streamlit Web Application

A user-friendly interface was built using Streamlit to make the model accessible without requiring API knowledge. Users input applicant details through an interactive form and receive real-time default risk predictions with color-coded risk tiers.

**Live Application:** https://loan-default-prediction-aryanparija.streamlit.app/

### Running Locally

```bash
cd app
streamlit run streamlit_app.py
```

The application will open at `http://localhost:8501`

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.12 |
| ML Framework | LightGBM, Scikit-learn, XGBoost (comparison) |
| Imbalance Handling | `scale_pos_weight` (LightGBM native) |
| Explainability | SHAP `TreeExplainer` |
| API Framework | FastAPI |
| UI Framework | Streamlit |
| Server | Uvicorn |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Model Serialization | Joblib |
| Deployment | Streamlit Community Cloud |

---

## Project Structure

```
loan-default-prediction/
├── data/
│   ├── application_train.csv
│   ├── bureau.csv
│   ├── previous_application.csv
│   └── HomeCredit_columns_description.csv
├── notebooks/
│   ├── 01_eda.ipynb          ← application + bureau + previous_application EDA
│   └── 02_modeling.ipynb     ← aggregation, preprocessing, training, GridSearchCV, SHAP
├── models/
│   ├── loan_default_model.pkl
│   ├── categorical_cols.pkl
│   └── numerical_cols.pkl
├── app/
│   ├── main.py               ← FastAPI app
│   └── streamlit_app.py      ← Streamlit UI
├── requirements.txt
└── README.md
```

---

## Author

**Aryan Parija**
[LinkedIn](https://www.linkedin.com/in/aryanparija2006/) | [GitHub](https://github.com/aryanparija)