# Repository audit

Audit date: 2026-10-01. Raw datasets were inspected read-only and in chunks.

## Baseline and observed data

- The current `creditforyou` CLI passes 17 tests. The legacy Streamlit source compiles, but its
  environment is not reproducible from the root requirements (`pypdf` is absent) and it calls the
  archived synthetic-target model. It is therefore a UI reference, not the final runtime.
- Synthetic banking: 500 customers, 1,426 accounts, 233 loans and 1,669,697 transactions. Transaction
  dates cover 2021-10-01 through 2026-10-01. No key or transaction-ID integrity errors were observed.
- Lending Club: 753,566 rows, issued 2007-06 through 2018-12. Final outcomes observed are 358,329
  `Fully Paid`, 90,137 `Charged Off`, 13 `Default`, plus 916 legacy-policy final outcomes. Open or
  unresolved statuses are excluded from baseline training.

## Implementation checklist

| Plan requirement | Existing component | Action |
|---|---|---|
| CLI/package/config | `creditforyou/`, `config/` | KEEP and extend |
| Memory-safe synthetic ingestion and join checks | none | CREATE `datasets.py` and scripts |
| Transaction classification and cash-flow features | `income.py` partial | MODIFY |
| Income evaluation against held-out ground truth | KPI fragment in `report.py` | CREATE dedicated evaluator/report |
| Actual Lending Club outcome target | generic `parse_label` | REPLACE with explicit target builder |
| Split-before-fit model pipeline | `model.py` already uses sklearn pipeline | KEEP and strengthen |
| Leakage guard and complete risk metrics | partial | MODIFY |
| Income + debt + PD integration | `pipeline.py` partial | MODIFY using synthetic `loans.csv` |
| Missing scoring components | currently fabricated neutral/zero values | REPLACE with explicit weight normalization |
| Streamlit dashboard | `legacy/app.py` | CREATE focused root `app.py`; legacy remains archived |
| SQLite persistence | `legacy/db.py` | CREATE focused root `db.py` |
| Explainability | legacy SHAP tied to old model | REPLACE with model-agnostic contribution utility |
| Tests and documentation | partial | MODIFY/CREATE |
| Synthetic target and old artifacts | under `legacy/` | REMOVE from final pipeline; retain archive only |

## Necessary schema adjustment

Synthetic customers do not contain Lending Club bureau variables such as FICO, delinquencies or
inquiries. The MVP model therefore uses only fields that can be derived or requested at inference
(loan/income ratios, debt ratio, term, employment tenure, home ownership and purpose). Transaction
features remain in the final hybrid score and are not falsely presented as Lending Club model inputs.
