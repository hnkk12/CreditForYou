# CreditForYou

CreditForYou is an academic prototype that combines transaction-based income estimation, a historical
loan default model and a transparent hybrid score. It is not production-ready, is not a bank policy,
and is not validated for Vietnamese borrowers.

## Architecture and data separation

```text
Synthetic banking transactions ─> income/cash-flow features ─┐
                                                              ├─> score 0–100 ─> Tier A/B/C/D ─> recommendation
Lending Club resolved outcomes ─> PD model ───────────────────┘
Synthetic active loans ─> current monthly debt ───────────────┘
```

The populations remain separate. Lending Club borrowers are never joined to synthetic customers by
ID. `customers.monthly_income` is used only after prediction to evaluate the estimator; it is never an
estimator input or the income passed to the main demo flow.

- Synthetic banking data is used for transaction analysis, income estimation, stability, confidence,
  expense, balance and cash-flow behaviour. It contains USD-denominated synthetic values.
- US Lending Club data is used only for actual historical loan outcomes and PD training. It does not
  directly represent a Vietnamese banking population.
- Transaction features do not enter the Lending Club model because no dataset here contains both
  transaction history and actual default outcome for the same population. They enter the final hybrid
  scoring layer instead.

## Repository

```text
app.py                         Streamlit dashboard
db.py                          SQLite assessment persistence
creditforyou/
  datasets.py                  discovery, chunked ingestion, join validation
  income.py                    transaction classification and cash-flow features
  lendingclub.py               target mapping and leakage guard
  model.py                     split-safe sklearn training and inference
  scoring.py                   normalized hybrid scoring and reason codes
  pipeline.py                  integrated generic and synthetic-customer flows
  evaluation.py                actual income metrics
config/                        schema, thresholds and weights
scripts/                       validation, feature building, evaluation and training
reports/                       generated metrics and predictions
models/                        generated final model artifact
tests/                         unit and end-to-end tests
legacy/                        archived AltCredict UI/model; not called by final pipeline
```

## Setup and reproducible run

Python 3.12 is recommended.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

python scripts/validate_data.py
python scripts/evaluate_income_estimation.py
python scripts/train_credit_model.py --max-rows 300000
python -m pytest -q
streamlit run app.py
```

The supplied paths are detected under `dataset isb/`. Raw CSV files are read-only. Generated caches go
to `data/processed/`, model artifacts to `models/`, and evaluation evidence to `reports/`.

CLI compatibility remains available:

```powershell
python -m creditforyou --help
python -m creditforyou income -t "dataset isb/data/transactions.csv" -c C0000001
python -m creditforyou train -l "dataset isb/accepted_2007_to_2018Q4.csv" --max-rows 300000
```

## Income module

Classification prioritizes `is_salary`, `is_transfer`, `category`, `subcategory`, `label` and
`transaction_type`. Internal transfers, refunds/reversals, loan disbursements and other configured
non-income inflows are excluded. Generic bank statements without rich flags use a documented fallback
based on external positive credits and exclusion keywords.

Output features include:

- active months and income transaction count;
- average, median, minimum, maximum and standard deviation of monthly income;
- income CV, heuristic stability score, estimated income and evidence-based confidence;
- recurring-income ratio and income-source count;
- average expense, expense/income ratio and monthly net cash flow;
- positive cash-flow month ratio, average/minimum balance and cash-flow volatility.

The default estimate uses the configured 10th percentile of positive recurring monthly income over
up to 12 months, with at least 6 months preferred when available. Salary is accepted explicitly; non-salary external credit must recur from the same
source. Transfers, refunds, loan disbursements, anomalous one-time credits, zero-income months and
partial boundary months do not inflate the estimate. This is a prototype heuristic, not verified income.

### Executed income evaluation

The committed `reports/income_metrics.json` was produced over all 500 synthetic customers and
1,669,697 transactions:

| Metric | Result |
|---|---:|
| MAE | 877.00 USD |
| RMSE | 1,724.37 USD |
| MAPE | 22.89% |
| sMAPE | 27.18% |
| Median absolute percentage error | 8.83% |
| Within ±10% | 61.60% |
| Within ±20% | 72.40% |

The proposed “90% within ±10%” KPI is still **not achieved**. Mean signed error is -50.24 USD,
consistent with the intentionally conservative lower-tail estimate.

## Credit-risk model

`loan_status` mapping is explicit:

- `Fully Paid` → 0;
- `Charged Off` and `Default` → 1;
- `Current`, late, grace-period and other unresolved/non-baseline statuses → excluded.

The final core model features are `loan_to_income`, `installment_to_income`, `dti`,
`years_experience`, `term_months`, `log_annual_income`, `log_loan_amount`, `home_ownership`, `purpose`
and `verification_status`. Synthetic transaction-derived income is explicitly marked `not_verified`. These can be derived from the income
module/current debt or requested in the UI. Bureau fields were excluded from the MVP because the
synthetic applicant population cannot supply them. The model uses annualized estimated income only via
unit-invariant ratios; it never receives synthetic income ground truth.

The guard forbids platform-derived or post-loan fields including `grade`, `sub_grade`, `int_rate`,
`installment`, `out_prncp*`, `total_pymnt*`, `total_rec_*`, `recoveries`, payment-after-origination,
last-FICO, `hardship_*`, `debt_settlement_*` and `settlement_*` columns.

Preprocessing is inside each sklearn pipeline and is fit only on the training split. Data is divided
60%/20%/20% into train/validation/test. Logistic, class-weighted Logistic, HistGradientBoosting,
class-weighted HistGradientBoosting and class-weighted Random Forest are compared on validation only;
the threshold maximizes validation F1. The test set is evaluated once after model and threshold selection.

### Executed credit evaluation

The model was trained with a fixed seed on 300,000 randomly sampled resolved outcomes; the holdout has
60,000 rows and a 20.14% default rate. Selected Gradient Boosting results:

| Metric | Result |
|---|---:|
| ROC-AUC | 0.6667 |
| PR-AUC | 0.3349 |
| Gini / KS | 0.3334 / 0.2388 |
| Brier score | 0.1512 |
| Validation-selected threshold | 0.2067 |
| Precision / Recall / F1 | 0.3010 / 0.5745 / 0.3951 |
| Confusion matrix | `[[31798, 16118], [5142, 6942]]` |

Calibration-bin evidence and Logistic Regression comparison are in
`reports/credit_model_metrics.json`. Accuracy is not used as the primary metric.

## Integration and scoring

At inference, `estimated_monthly_income × 12` is the conceptual Lending Club annual-income mapping;
the implementation converts it into unit-free loan/income and payment/income ratios. Synthetic active
loan payments form `existing_monthly_debt`. The affordability measure is:

```text
post_loan_debt_ratio = (existing_monthly_debt + requested_monthly_installment)
                       / estimated_monthly_income
```

All ratios are 0–1 (not percentages) internally. Default weights are income stability 25,
affordability 25, repayment/credit history 20, tenure/experience 15 and PD 15. If evidence is missing,
the component is disabled and available weights are normalized to 100; a neutral/random value is not
invented. Tier thresholds and business rules are configurable in `config/scoring.json`.

Auto-approval is blocked when requested installment exceeds 30% of estimated income, total post-loan
debt service exceeds 45%, expenses exceed 90%, PD exceeds 25%, or income confidence is below 70%.
Hard rejection applies above 70% total debt service or 120% expense/income. These are documented,
configurable research guardrails rather than validated bank policy.

The UI labels PD as “model-predicted default probability,” shows component breakdown and reason codes,
and exposes local missing-feature sensitivity rather than claiming causal SHAP explanations.

## Limitations

- The income data is synthetic and the PD data is historical US Lending Club data.
- PD and transaction modules are trained/evaluated on different populations.
- No transaction feature has an end-to-end default label in this repository.
- The score, tier, interest range and recommendation are configurable prototype heuristics.
- PD calibration and discrimination must be revalidated on the intended deployment population.
- The income estimator improves materially but still misses the proposed 90% within-±10% KPI.
- The stored artifact represents a reproducible 300,000-row baseline, not an exhaustive model search.
