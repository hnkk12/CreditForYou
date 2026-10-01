# Definition of Done verification

Verified on 2026-10-01.

| Requirement | Status | Implementation | Evidence / test |
|---|---|---|---|
| Select a synthetic customer | DONE | Streamlit selector in `app.py` | Headless app startup; integrated `C0000001` smoke run |
| Read selected transaction history | DONE | Chunk-filtered `datasets.load_transactions` | 1,669,697-row integrity scan; UI customer cache |
| Estimate income without ground truth input | DONE | `income.prepare_transactions` / `estimate_income` | Evaluator passes transactions alone; 21-test suite |
| Stability and confidence | DONE | Configured heuristic and coverage/quality confidence | Unit tests and `income_predictions.csv` |
| Ground-truth comparison report | DONE | `evaluation.evaluate_income_dataset` | `income_metrics.json`, 500 customers |
| Actual Lending Club outcome model | DONE | Explicit `loan_status` target | `credit_model_metrics.json`, 300,000 resolved outcomes |
| No post-loan leakage | DONE | `LEAKAGE_COLUMNS` and fail-fast guard | `test_leakage_guard` |
| Split before preprocessing fit | DONE | sklearn pipelines fitted after split | `model.train`; end-to-end model test |
| PD output | DONE | `predict_proba` | range assertion in pipeline test |
| Income-to-credit mapping | DONE | `annual_inc_for_model = estimate × 12`, unit-free ratios | integrated customer smoke output |
| Final score 0–100 | DONE | normalized available-component scoring | boundary/unit tests |
| Tier and decision | DONE | config-driven A/B/C/D and decision rules | tier-boundary tests |
| UI breakdown and reasons | DONE | Streamlit component chart/reason codes | headless Streamlit startup |
| Provenance and two-population limitation | DONE | README architecture and limitations | README review |
| Clean-environment instructions | PARTIAL | pinned minimum dependencies and ordered commands | commands pass in current Python 3.12 environment; a second isolated environment was not created |

The MVP is functional. The last item remains PARTIAL only because dependency installation was not repeated in a newly
created isolated environment; no implementation component is silently marked complete on that basis.
