# Loan Default Prediction: Classical ML vs. Deep Tabular Learning on Imbalanced Multi-Relational Data

A production-grade machine learning system and empirical benchmark for predicting consumer loan default risk across all **7 relational tables** (**307,511 primary loan applications** linked to **58.4M+ auxiliary behavioral records**) from the [Home Credit Default Risk](https://www.google.com/search?q=https://www.kaggle.com/competitions/home-credit-default-risk) database.

The system benchmarks **Gradient-Boosted Decision Trees (LightGBM, CatBoost, XGBoost)** and classical baselines against **modern Deep Tabular Architectures (Tabular ResNet, Multi-Layer Perceptron, FT-Transformer)** under a strictly **leakage-free, tuning-matched protocol**, and serves real-time predictions with **post-hoc probability calibration**, **decision-theoretic cost thresholding**, **TreeSHAP explainability**, and **demographic fairness auditing** via a **FastAPI** backend and **Streamlit** web application.

🔗 **Live Demo:** [https://loan-default-prediction-aryanparija.streamlit.app/](https://loan-default-prediction-aryanparija.streamlit.app/)

## Key Highlights & Benchmark Contributions

1. **Full 7-Table Relational Feature Engineering (147 Raw -> 261 Encoded Features)**:
   - Aggregates **58.4M+ rows** across external credit bureau histories (`bureau.csv`, `bureau_balance.csv`), prior internal loans (`previous_application.csv`), historical repayment timeliness and delinquency velocity (`installments_payments.csv`), monthly point-of-sale contracts (`POS_CASH_balance.csv`), and revolving credit utilization (`credit_card_balance.csv`).
   - Progressive ablation quantifies the exact lift across relational stages: **1-Table Baseline (`0.7570` ROC-AUC)** -> **3-Table Untuned (`0.7640`)** -> **3-Table Tuned (`0.7696`)** -> **7-Table Full Benchmark (`0.7876` ROC-AUC, `0.2870` PR-AUC)**.

2. **Tuning-Matched & Sample-Matched Cross-Paradigm Benchmark**:
   - **Tuned LightGBM** achieves the highest standalone test discrimination (**`0.7876` ROC-AUC**, **`0.2837` PR-AUC**, **`0.303` F1**, 5-Fold Stratified CV **`0.7828 ± 0.0020`**), followed by **CatBoost (`0.7834`)** and **XGBoost (`0.7829`)**.
   - Remarkable finding: A **7-Table Tuned Logistic Regression (`0.7704` ROC-AUC)** outperforms the **3-Table Tuned LightGBM (`0.7696`)**, demonstrating that domain-informed multi-table feature engineering outweighs algorithm complexity.
   - Among deep tabular models evaluated on an identical sample-matched split, a 3-seed **Tabular ResNet (`0.7799` ROC-AUC)** outperforms a 3-seed **PyTorch MLP (`0.7785`)** and **FT-Transformer (`0.7763`)**, while training 23x–88x slower than LightGBM.
   - A **Rank-Averaged Hybrid Ensemble (`LightGBM + Tabular ResNet + MLP`)** reaches **`0.7876` ROC-AUC (`95% DeLong CI: [0.7811, 0.7942]`)** and **`0.2870` PR-AUC**, achieving a statistically significant lift over standalone LightGBM on the matched split (**Paired DeLong `z = 2.323, p = 0.0202 < 0.05`**).

3. **Empirical Proof of the Pre-Split SMOTE Data Leakage Trap**:
   - Proves experimentally that applying SMOTE *before* `train_test_split` (a common flaw in recent papers claiming >0.97 AUC) fabricates an illusory **`0.9809` ROC-AUC** via synthetic nearest-neighbor leakage across train and test sets, whereas strictly leakage-free in-fold SMOTE drops to **`0.7473`**, underperforming native cost-sensitive loss weighting (`scale_pos_weight = 11`).

4. **156x Reduction in Calibration Error (ECE) & Decision-Theoretic Thresholding**:
   - Post-hoc **Isotonic Regression calibration** corrects the probability shift induced by `scale_pos_weight = 11` (which inflated mean predicted PD to `0.3630` vs. the true `0.0807` default rate), slashing **Expected Calibration Error (ECE) by 156-fold (`0.2822` -> `0.0018`)** and **Brier Score from `0.1666` to `0.0659`**.
   - Derives optimal approval thresholds (`t*`) under asymmetric False Negative to False Positive cost ratios: **5:1 (`t* = 0.175`, F1 = `0.338`)**, **10:1 (`t* = 0.080`, Recall = `0.681`)**, and **15:1 (`t* = 0.065`, Recall = `0.791`)**.

5. **TreeSHAP Interpretability & Demographic Fairness Audit**:
   - Audits subgroup discrimination and error parity across `CODE_GENDER` and `AGE_YEARS`, and proves via a **Fairness-Blind Ablation** that completely removing gender and age reduces test ROC-AUC by only **`-0.0035` (`0.7876` -> `0.7841`)**.

## Multi-Relational Dataset Architecture (All 7 Tables)

| Table Name | Granularity & Join Key | Raw Records | Engineered Aggregates & Key Signals |
| :--- | :--- | :--- | :--- |
| **`application_train.csv`** | 1 row per current loan (`SK_ID_CURR`) | 307,511 × 122 | Demographics, affordability ratios, external bureau consensus (`EXT_SOURCE_MEAN`, `EXT_SOURCE_STD`), social circle DPD |
| **`bureau.csv`** | 1 row per external CB loan (`SK_ID_CURR`) | 1,716,428 × 17 | Active/total loans, credit type diversity, credit age (`MEAN`/`MIN`), overdue amounts, `BUREAU_DEBT_CREDIT_RATIO` |
| **`bureau_balance.csv`** | Monthly CB status (`SK_ID_BUREAU` -> `SK_ID_CURR`) | 27,299,925 × 3 | Total tracked months, historical 1–5 DPD month counts, mean/max DPD frequency (`BUREAU_BB_DPD_RATE_MEAN`) |
| **`previous_application.csv`** | 1 row per prior HC loan (`SK_ID_CURR`) | 1,670,214 × 37 | Prior application/approval/refusal/cancellation counts, decision recency, `PREV_REFUSAL_RATE`, `PREV_CREDIT_APPLICATION_RATIO` |
| **`installments_payments.csv`** | 1 row per payment (`SK_ID_CURR`) | 13,605,401 × 8 | Days Past Due (`DPD`), Days Before Due (`DBD`), `INSTAL_LATE_RATE`, payment shortfall, 1-year delinquency acceleration (`INSTAL_RECENT_1Y_LATE_RATE`) |
| **`POS_CASH_balance.csv`** | Monthly POS/cash snapshot (`SK_ID_CURR`) | 10,001,358 × 8 | Active/completed POS contracts (`POS_COMPLETED_RATIO`), mean/max `SK_DPD`, future remaining installments |
| **`credit_card_balance.csv`** | Monthly revolving CC log (`SK_ID_CURR`) | 3,840,312 × 23 | Revolving limit utilization (`CC_UTILIZATION_MEAN`, `MAX`), missed minimum payment rate (`CC_MIN_PAY_MISSED_RATE`), CC DPD |

- **Target Distribution**: `91.93%` Repaid (`0`) vs. `8.07%` Defaulted (`1`) — an `11.39 : 1` imbalance ratio.
- **Validation Protocol**: Stratified 80/20 train/test split (`246,008` train / `61,503` test with `4,965` defaulters, `random_state = 42`), plus 5-Fold Stratified Cross-Validation on the training split and a 10% validation slice (`24,601` rows) for neural early stopping.

## Experimental Results

### 1. Progressive Multi-Table Ablation Study (LightGBM on Held-Out Test Set, N = 61,503)

| Ablation Stage | Relational Tables Included | Raw / Encoded Features | Test ROC-AUC | Test PR-AUC | Minority F1 (t=0.5) | Incremental Gain |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Stage 1: Baseline** | 1 Table (`application_train` only) | 75 / 190 | `0.7570` | — | `0.270` | Baseline |
| **Stage 2: 3-Table Untuned** | 3 Tables (`+ bureau, previous_application`) | 102 / 214 | `0.7640` | — | `0.274` | `+0.0070` AUC |
| **Stage 3: 3-Table Tuned** | 3 Tables + GridSearchCV Tuning | 102 / 214 | `0.7696` | `0.2634` | `0.279` | `+0.0056` AUC |
| **Stage 4: 7-Table Full Benchmark** | All 7 Tables (`+ BB, Instal, POS, CC, EXT_1`) | **147 / 261** | **`0.7876`** | **`0.2837`** | **`0.303`** | **`+0.0180` AUC** |

### 2. Tuning-Matched Classical & GBDT Benchmark (Full 7-Table Feature Set)

| Model Configuration | 5-Fold CV ROC-AUC (Mean ± Std) | 5-Fold 95% CI | Test ROC-AUC | Test PR-AUC | F1 (t=0.5) | Raw Brier Score | Tune + Fit Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM (Tuned 7-Table)** | **`0.7828 ± 0.0020`** | **`[0.7810, 0.7845]`** | **`0.7876`** | **`0.2837`** | **`0.303`** | `0.1675` | `133.4s` |
| **CatBoost (Tuned 7-Table)** | — | — | **`0.7834`** | **`0.2791`** | `0.292` | `0.1857` | `339.4s` |
| **XGBoost (Tuned 7-Table)** | `0.7785 ± 0.0017` | `[0.7770, 0.7799]` | **`0.7829`** | **`0.2774`** | `0.297` | `0.1740` | `117.1s` |
| **Random Forest (Tuned 7-Table)** | — | — | `0.7704` | `0.2532` | **`0.311`** | **`0.1539`** | `513.3s` |
| **Logistic Regression (Tuned)** | `0.7672 ± 0.0018` | `[0.7656, 0.7688]` | `0.7704` | `0.2569` | `0.276` | `0.1942` | **`27.8s`** |

### 3. Sample-Matched GBDT vs. Modern Deep Tabular Learning & Paired DeLong Tests

Evaluated on the identical 90% training (`221,407` rows) / 10% early-stopping validation (`24,601` rows) split and scored once on the held-out test set (`61,503` rows):

| Rank & Model Architecture | Paradigm | Test ROC-AUC | DeLong 95% CI | Test PR-AUC | Delta vs. 7-Tab LGBM | DeLong z-stat | DeLong p-value | Fit Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Hybrid Blend (LGBM + ResNet + MLP)** | Multi-Paradigm Ensemble | **`0.7876`** | **`[0.7811, 0.7942]`** | **`0.2870`** | **`+0.0015`** | **`+2.323`** | **`0.0202*`** | `889.6s` |
| **2. Tuned LightGBM (7-Table)** | Leaf-Wise GBDT | **`0.7862`** | `[0.7796, 0.7927]` | **`0.2832`** | `0.0000` | `0.000` | `1.0000` | **`11.9s`** |
| **3. Tabular ResNet (3-Seed Ensemble)** | Residual Deep Net | **`0.7799`** | `[0.7732, 0.7865]` | `0.2766` | `-0.0063` | `-4.474` | `7.67e-06***` | `597.2s` |
| **4. PyTorch MLP (3-Seed Ensemble)** | Feed-Forward Deep Net | `0.7785` | `[0.7718, 0.7852]` | `0.2780` | `-0.0077` | `-5.258` | `1.46e-07***` | `280.5s` |
| **5. FT-Transformer (Attention)** | Self-Attention Tabular Net | `0.7763` | `[0.7696, 0.7830]` | `0.2675` | `-0.0099` | `-6.298` | `3.02e-10***` | `1056.5s` |

### 4. Empirical Proof of the Pre-Split SMOTE Resampling Leakage Trap

| Imbalance Protocol | Test Set Distribution | Reported Test ROC-AUC | Methodological Verdict |
| :--- | :--- | :---: | :--- |
| **1. Pre-Split SMOTE (Flawed — Data Leakage)** | Artificially 50% / 50% Balanced (Leaked) | **`0.9809`** | **Invalid** (Synthetic nearest-neighbor test leakage) |
| **2. In-Fold SMOTE (Leakage-Free, 30k Sample)** | Natural 91.93% / 8.07% Imbalance | `0.7473` | **Valid** (Synthetic interpolation noise in 261-D space) |
| **3. Cost-Sensitive Weighting (Full 7-Table)** | Natural 91.93% / 8.07% Imbalance | **`0.7876`** | **Optimal** (Preserves exact feature geometry) |

### 5. Post-Hoc Probability Calibration & Decision-Theoretic Cost Thresholds

| Calibration / Cost Scenario | Test ROC-AUC | Brier Score (Down) | ECE (Down) | Mean Pred. PD (True = `0.0807`) | Optimal Threshold (t*) | Recall at t* | Precision at t* | F1 at t* |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Uncalibrated LightGBM (`w+ = 11`)** | `0.7862` | `0.1666` | `0.2822` | `0.3630` (Distorted) | `t = 0.500` | `0.677` | `0.189` | `0.303` |
| **Platt Scaling (Sigmoid Calibrated)** | `0.7862` | `0.0661` | `0.0044` | **`0.0795`** | — | — | — | ECE reduced 64x |
| **Isotonic Regression Calibrated** | `0.7857` | **`0.0659`** | **`0.0018`** | **`0.0796`** | — | — | — | **ECE reduced 156x** |
| **Cost Ratio FN:FP = 5:1 (Moderate)** | `0.7857` | `0.0659` | `0.0018` | `0.0796` | **`t* = 0.175`** (Bayes `0.167`) | `0.395` | **`0.295`** | **`0.338`** |
| **Cost Ratio FN:FP = 10:1 (Standard)** | `0.7857` | `0.0659` | `0.0018` | `0.0796` | **`t* = 0.080`** (Bayes `0.091`) | **`0.681`** | `0.193` | `0.301` |
| **Cost Ratio FN:FP = 15:1 (Conservative)** | `0.7857` | `0.0659` | `0.0018` | `0.0796` | **`t* = 0.065`** (Bayes `0.062`) | **`0.791`** | `0.159` | `0.264` |

### 6. Demographic Fairness Audit & Protected-Attribute Ablation

| Subgroup / Ablation | Test Count (N) | Actual Default Rate | Subgroup ROC-AUC | TPR (Default Recall) | FPR (False Alarm) | Approval Rate (`Pred=0`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gender: Female (`F`)** | 40,561 | `6.99%` | `0.7805` | `0.6241` | `0.2069` | `76.39%` |
| **Gender: Male (`M`)** | 20,940 | `10.17%` | **`0.7884`** | `0.7586` | `0.3284` | `62.79%` |
| **Age: Young (`< 25 yrs`)** | 2,385 | `11.74%` | `0.7338` | `0.8286` | `0.5021` | `45.95%` |
| **Age: Prime (`25–50 yrs`)** | 38,253 | `9.04%` | **`0.7901`** | `0.7256` | `0.2867` | `67.36%` |
| **Age: Senior (`> 50 yrs`)** | 20,865 | `5.88%` | `0.7702` | `0.5249` | `0.1502` | `82.77%` |
| **Fairness-Blind LightGBM (No Gender/Age)** | 61,503 | `8.07%` | **`0.7841`** (`Delta = -0.0035`) | — | — | Retains **99.56%** of full AUC |

## Comparison with Published Open-Access Literature (2019–2026)

| Study & Venue | Tables Used | Best Reported Score | This Work (**0.7876** ROC-AUC / **0.2870** PR-AUC) |
| :--- | :---: | :--- | :--- |
| [Ben Ammar \& Rubaidi (2026)](https://www.google.com/search?q=https://www.mdpi.com/2227-7390/14/9/1467), *MDPI Mathematics* | 1 table | GBDT `0.7498` AUC (`0.2118` PR-AUC, `0.254` F1) | **Outperforms by `+0.0378` ROC-AUC, `+0.0752` PR-AUC, and `+0.084` F1** |
| [Yang \& Vasistha (2026)](https://www.google.com/search?q=https://arxiv.org/html/2601.14633v1), *arXiv (UIUC)* | 6 tables | Standalone LGBM `0.7690`; Hybrid GNN+LGBM `0.7816` (`0.2807` PR-AUC) | **Outperforms both standalone LGBM (`+0.0186`) and Hybrid GNN+LGBM (`+0.0060` AUC, `+0.0063` PR-AUC)** |
| [Tanna et al. (2026)](https://www.google.com/search?q=https://arxiv.org/html/2505.18635v1), *arXiv (Lexsi Labs)* | 7 tables | Classical RF `0.739`, LGBM `0.713`; TabPFN (50K) `0.7860` (`0.302` F1) | **Outperforms classical baselines by `+0.048+` and edges out TabPFN (`0.7876` vs. `0.7860` AUC, `0.338` vs. `0.302` F1)** |
| [Hlongwane et al. (2024b)](https://www.google.com/search?q=https://pmc.ncbi.nlm.nih.gov/articles/PMC11318906/), *PLOS ONE* | 7 tables -> 11 vars | XGBoost `0.6977` AUC, LightGBM `0.6965` AUC | **Outperforms by `+0.0900` ROC-AUC** |
| [Hlongwane et al. (2024a)](https://www.google.com/search?q=https://pmc.ncbi.nlm.nih.gov/articles/PMC11108212/), *PLOS ONE* | 7 tables | LightGBM `0.7936`, XGBoost `0.7892`, CatBoost `0.7890` | Comparable 7-table GBDT ceiling while adding **3 Deep Learning families, PR-AUC, DeLong tests, 156x ECE calibration, and fairness** |
| [Qiu et al. (2019)](https://www.google.com/search?q=https://www.researchgate.net/publication/338526492_Credit_Risk_Scoring_Analysis_Based_on_Machine_Learning_Models), *IEEE ICISCE* | 3 tables | Original LGBM `0.721`; Expert + Pruning `0.755 / 0.778*` | **Outperforms across LR (`0.7704` vs. `0.711`), RF (`0.7704` vs. `0.684`), and LightGBM (`0.7876` vs. `0.778`)** |
| [IJERT (2026)](https://www.google.com/search?q=https://www.ijert.org/an-explainable-and-fair-machine-learning-approach-for-credit-default-prediction-ijertv15is100217) | 1 table | Tuned XGBoost `0.7570` AUC | **Outperforms by `+0.0306` ROC-AUC** and resolves MLP minority collapse (`0.7785` MLP AUC) |

## Project Structure

```text
Loan-Default-Prediction/
├── data/                              # Raw Kaggle CSVs (git-ignored)
│   ├── application_train.csv
│   ├── bureau.csv
│   ├── bureau_balance.csv
│   ├── previous_application.csv
│   ├── installments_payments.csv
│   ├── POS_CASH_balance.csv
│   └── credit_card_balance.csv
├── notebooks/
│   ├── 01_eda.ipynb                   # 1-table, 3-table, and 7-table relational EDA & correlations
│   ├── 02_modeling.ipynb              # Progressive ablation, equal-budget GBDT tuning, 5-fold CV, SMOTE leakage proof, SHAP & fairness
│   └── 03_mlp_comparison.ipynb        # PyTorch MLP, Tabular ResNet, FT-Transformer, Rank Blend, DeLong tests, Calibration & Cost curves
├── models/
│   ├── loan_default_model.pkl         # Serialized preprocessing + LightGBM pipeline
│   ├── categorical_cols.pkl
│   └── numerical_cols.pkl
├── app/
│   ├── main.py                        # FastAPI REST backend (raw & calibrated PD + cost-sensitive thresholds)
│   └── streamlit_app.py               # Interactive Streamlit underwriting UI (with Bureau & Prior History inputs)
├── requirements.txt
└── README.md
```

## How to Reproduce & Run Locally

1. **Download the 7 CSV tables** from [Kaggle Home Credit Default Risk](https://www.google.com/search?q=https://www.kaggle.com/competitions/home-credit-default-risk/data) into `data/`.
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the experimental notebooks in order**:
   - `notebooks/01_eda.ipynb`
   - `notebooks/02_modeling.ipynb`
   - `notebooks/03_mlp_comparison.ipynb`
4. **Launch the FastAPI Backend**:
   ```bash
   cd app
   uvicorn main:app --reload
   ```
   Open Swagger UI at `http://127.0.0.1:8000/docs`.
5. **Launch the Streamlit Web Application**:
   ```bash
   cd app
   streamlit run streamlit_app.py
   ```

## Authors

- **Aryan Parija** 
 [LinkedIn](https://www.google.com/search?q=https://www.linkedin.com/in/aryanparija2006/) | [GitHub](https://github.com/aryanparija)