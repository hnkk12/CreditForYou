"""Mô hình ML xác suất vỡ nợ (PD) — thành phần 5 của điểm tín dụng.

Huấn luyện trên dataset khoản vay có nhãn (VD: Lending Club), áp dụng cho khách CreditForYou.
Để mô hình "chuyển" được giữa 2 nguồn dữ liệu (USD/năm vs VND/tháng), chỉ dùng đặc trưng
KHÔNG PHỤ THUỘC ĐƠN VỊ TIỀN (tỷ lệ) + đặc trưng mà hồ sơ khách VN cũng có.

Chống rò rỉ dữ liệu (leakage):
- Không dùng cột sinh ra SAU khi cho vay (total_pymnt, recoveries, last_pymnt_d, out_prncp...).
- Không dùng grade / sub_grade / int_rate (là đầu ra mô hình rủi ro của Lending Club).
- Không dùng 'installment' gốc của Lending Club (ngầm chứa lãi suất -> grade). Khoản trả hằng tháng
  được TÍNH LẠI bằng lãi suất cố định trong config cho cả lúc train và lúc chấm điểm.
- Không dùng class_weight='balanced' để PD giữ đúng ý nghĩa xác suất.
"""

from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .config import DEFAULT_MODEL_PATH, scoring_config
from .io import load_dataset, normalize_name, parse_amount, parse_percent_ratio, \
    parse_term_months, parse_years
from .lendingclub import assert_no_leakage, build_default_target, target_counts
from .scoring import annuity_payment

CORE_NUM = ["loan_to_income", "installment_to_income", "dti", "years_experience", "term_months",
            "log_annual_income", "log_loan_amount"]
CORE_CAT = ["home_ownership", "purpose", "verification_status"]
BUREAU_NUM = ["delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec", "revol_util", "age"]


# --------------------------------------------------------------------------- đặc trưng

def _norm_home(s: pd.Series) -> pd.Series:
    m = {"own": "own", "owner": "own", "so_huu": "own", "nha_rieng": "own",
         "mortgage": "mortgage", "tra_gop": "mortgage",
         "rent": "rent", "thue": "rent", "nha_thue": "rent",
         "family": "family", "o_voi_gia_dinh": "family", "gia_dinh": "family"}
    return s.astype(str).map(normalize_name).map(lambda x: m.get(x, "other" if x not in ("nan", "none", "") else np.nan))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """df có cột chuẩn: requested_amount, monthly_income (VND hoặc USD - chỉ dùng tỷ lệ),
    term_months, years_experience, existing_monthly_debt hoặc dti, home_ownership, purpose..."""
    loan_cfg = scoring_config()["loan_defaults"]
    out = pd.DataFrame(index=df.index)
    income = pd.to_numeric(df.get("monthly_income"), errors="coerce")
    income = income.where(income > 0)
    amount = pd.to_numeric(df.get("requested_amount"), errors="coerce")
    term = pd.to_numeric(df.get("term_months"), errors="coerce") if "term_months" in df else \
        pd.Series(np.nan, index=df.index)

    out["loan_to_income"] = amount / (income * 12)
    out["log_annual_income"] = np.log1p((income * 12).clip(lower=0))
    out["log_loan_amount"] = np.log1p(amount.clip(lower=0))
    std_installment = [annuity_payment(a, loan_cfg["annual_interest_rate"], t if t == t else loan_cfg["term_months"])
                       if a == a else np.nan for a, t in zip(amount, term)]
    out["installment_to_income"] = pd.Series(std_installment, index=df.index) / income
    if "dti" in df:
        out["dti"] = pd.to_numeric(df["dti"], errors="coerce")
    elif "existing_monthly_debt" in df:
        out["dti"] = pd.to_numeric(df["existing_monthly_debt"], errors="coerce") / income
    else:
        out["dti"] = np.nan
    yrs = pd.to_numeric(df.get("years_experience"), errors="coerce") if "years_experience" in df else np.nan
    out["years_experience"] = np.minimum(yrs, 10)   # Lending Club ghi tối đa "10+ years"
    out["term_months"] = term
    out["home_ownership"] = _norm_home(df["home_ownership"]) if "home_ownership" in df else np.nan
    out["purpose"] = df["purpose"].astype(str).map(normalize_name).replace({"nan": np.nan, "": np.nan}) \
        if "purpose" in df else np.nan
    out["verification_status"] = (df["verification_status"].astype(str).map(normalize_name)
                                  .replace({"nan": np.nan, "": np.nan})
                                  if "verification_status" in df else np.nan)
    for c in BUREAU_NUM:
        out[c] = pd.to_numeric(df[c], errors="coerce") if c in df else np.nan
    # chặn giá trị cực đoan (thu nhập ~0 -> tỷ lệ vô hạn)
    for c in ("loan_to_income", "installment_to_income", "dti"):
        out[c] = out[c].replace([np.inf, -np.inf], np.nan).clip(upper=10)
    return out


# --------------------------------------------------------------------------- dữ liệu train

def load_training_data(path, max_rows=None, random_state=42):
    raw = load_dataset(path, "loans")
    target_info = target_counts(raw["default"])
    y = build_default_target(raw["default"])
    keep = y.notna()
    dropped = int((~keep).sum())
    raw, y = raw[keep], y[keep].astype(int)
    if max_rows and len(raw) > max_rows:
        idx = raw.sample(n=max_rows, random_state=random_state).index
        raw, y = raw.loc[idx], y.loc[idx]

    df = pd.DataFrame(index=raw.index)
    df["requested_amount"] = parse_amount(raw["requested_amount"])
    if "monthly_income" in raw:
        df["monthly_income"] = parse_amount(raw["monthly_income"])
    else:
        df["monthly_income"] = parse_amount(raw["annual_income"]) / 12
    if "term_months" in raw:
        df["term_months"] = parse_term_months(raw["term_months"])
    if "years_experience" in raw:
        df["years_experience"] = parse_years(raw["years_experience"])
    if "dti" in raw:
        df["dti"] = parse_percent_ratio(raw["dti"])
    if "revol_util" in raw:
        df["revol_util"] = parse_percent_ratio(raw["revol_util"])
    for c in ("home_ownership", "purpose", "verification_status", "delinq_2yrs", "inq_last_6mths",
              "open_acc", "pub_rec", "age"):
        if c in raw:
            df[c] = raw[c]
    info = {"rows_used": int(len(df)), "rows_dropped_unknown_label": dropped,
            "default_rate": float(y.mean()), "source_columns": list(raw.columns)}
    info["target_mapping"] = target_info
    return df, y, info


# --------------------------------------------------------------------------- train / predict

def _candidates(num, cat, random_state):
    lr = Pipeline([
        ("prep", ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                              ("sc", StandardScaler())]), num),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                              ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=20))]), cat),
        ])),
        ("clf", LogisticRegression(max_iter=2000, C=1.0)),
    ])
    def hgb(class_weight=None):
        return Pipeline([
        ("prep", ColumnTransformer([
            ("num", "passthrough", num),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                                   encoded_missing_value=np.nan), cat),
        ])),
        ("clf", HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=400, max_leaf_nodes=31, min_samples_leaf=50, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=30,
            categorical_features=list(range(len(num), len(num) + len(cat))), class_weight=class_weight,
            random_state=random_state)),
        ])
    lr_balanced = Pipeline([
        ("prep", clone(lr.named_steps["prep"])),
        ("clf", LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced")),
    ])
    rf = Pipeline([
        ("prep", ColumnTransformer([
            ("num", SimpleImputer(strategy="median", add_indicator=True), num),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                              ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), cat),
        ])),
        ("clf", RandomForestClassifier(
            n_estimators=250, max_depth=14, min_samples_leaf=30, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=-1, random_state=random_state)),
    ])
    return {"logistic_regression": lr, "logistic_regression_balanced": lr_balanced,
            "gradient_boosting": hgb(), "gradient_boosting_balanced": hgb("balanced"),
            "random_forest_balanced": rf}


def _best_f1_threshold(y_true, p):
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(y_true, p)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    idx = int(np.nanargmax(f1[:-1]))
    return float(thresholds[idx])


def classification_report_dict(y_true, p, threshold=None):
    y_true = np.asarray(y_true)
    auc = roc_auc_score(y_true, p)
    fpr, tpr, thr = roc_curve(y_true, p)
    ks_idx = int(np.argmax(tpr - fpr))
    best_thr = float(threshold if threshold is not None else thr[ks_idx])
    pred_05 = (p >= 0.5).astype(int)
    pred_best = (p >= best_thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred_best, labels=[0, 1]).ravel()
    prob_true, prob_pred = calibration_curve(y_true, p, n_bins=10, strategy="quantile")
    return {
        "roc_auc": round(auc, 4),
        "pr_auc": round(average_precision_score(y_true, p), 4),
        "gini": round(2 * auc - 1, 4),
        "ks": round(float(tpr[ks_idx] - fpr[ks_idx]), 4),
        "brier": round(brier_score_loss(y_true, p), 4),
        "accuracy@0.5": round(accuracy_score(y_true, pred_05), 4),
        "accuracy_baseline_all_good": round(1 - y_true.mean(), 4),
        "decision_threshold": round(best_thr, 4),
        "threshold_source": "validation_f1" if threshold is not None else "ks_on_same_data",
        "accuracy": round(accuracy_score(y_true, pred_best), 4),
        "precision": round(precision_score(y_true, pred_best, zero_division=0), 4),
        "recall": round(recall_score(y_true, pred_best, zero_division=0), 4),
        "f1": round(f1_score(y_true, pred_best, zero_division=0), 4),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "default_rate": round(float(y_true.mean()), 4),
        "mean_pd": round(float(np.mean(p)), 4),
        "calibration_bins": [
            {"mean_predicted_pd": round(float(pp), 4), "observed_default_rate": round(float(pt), 4)}
            for pp, pt in zip(prob_pred, prob_true)
        ],
    }


def train(path, feature_set="core", max_rows=None, test_size=0.2, validation_size=0.2, random_state=42,
          model_path=DEFAULT_MODEL_PATH, log=print):
    df, y, info = load_training_data(path, max_rows=max_rows, random_state=random_state)
    log(f"Dữ liệu train: {info['rows_used']:,} dòng | tỷ lệ vỡ nợ {info['default_rate']:.2%} | "
        f"bỏ {info['rows_dropped_unknown_label']:,} dòng chưa có kết cục")
    if y.nunique() < 2:
        raise ValueError("Nhãn chỉ có 1 lớp -> không train được. Kiểm tra cột default/loan_status.")

    X_all = build_features(df)
    num = CORE_NUM + (BUREAU_NUM if feature_set == "full" else [])
    cat = CORE_CAT
    # bỏ cột hoàn toàn trống trong dataset này
    empty = [c for c in num + cat if X_all[c].isna().all()]
    num = [c for c in num if c not in empty]
    cat = [c for c in cat if c not in empty]
    if empty:
        log(f"Bỏ đặc trưng không có dữ liệu: {empty}")
    assert_no_leakage(num + cat)
    X = X_all[num + cat].copy()
    X[cat] = X[cat].astype(object).where(X[cat].notna(), np.nan)

    X_dev, X_te, y_dev, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state)
    relative_val = validation_size / (1 - test_size)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_dev, y_dev, test_size=relative_val, stratify=y_dev, random_state=random_state)
    results, fitted, thresholds = {}, {}, {}
    for name, pipe in _candidates(num, cat, random_state).items():
        pipe.fit(X_tr, y_tr)
        p = pipe.predict_proba(X_val)[:, 1]
        thresholds[name] = _best_f1_threshold(y_val, p)
        results[name] = classification_report_dict(y_val, p, threshold=thresholds[name])
        fitted[name] = pipe
        log(f"  {name:30s} VAL AUC={results[name]['roc_auc']:.4f}  "
            f"PR-AUC={results[name]['pr_auc']:.4f}  F1={results[name]['f1']:.4f}")
    best = max(results, key=lambda k: (results[k]["roc_auc"], results[k]["pr_auc"]))

    # Test remains untouched until model and threshold are fixed on validation.
    final = fitted[best]
    test_probability = final.predict_proba(X_te)[:, 1]
    test_report = classification_report_dict(y_te, test_probability, threshold=thresholds[best])
    artifact = {
        "model": final, "model_name": best, "num_features": num, "cat_features": cat,
        "feature_set": feature_set, "metrics_validation": results,
        "metrics_test": {best: test_report}, "decision_threshold": thresholds[best], "train_info": info,
        "split_info": {"train_rows": int(len(X_tr)), "validation_rows": int(len(X_val)),
                       "test_rows": int(len(X_te)), "test_evaluated_only_after_selection": True},
        "trained_on": str(path), "trained_at": datetime.now().isoformat(timespec="seconds"),
        "model_version": "creditforyou-pd-v2", "target_definition": "resolved Lending Club loan_status",
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    log(f"Chọn trên validation: {best}; TEST AUC={test_report['roc_auc']:.4f}, "
        f"PR-AUC={test_report['pr_auc']:.4f}, F1={test_report['f1']:.4f} -> đã lưu {model_path}")
    return artifact


def load_model(model_path=DEFAULT_MODEL_PATH):
    if not model_path.exists():
        return None
    return joblib.load(model_path)


def predict_pd(artifact, features: pd.DataFrame) -> pd.Series:
    cols = artifact["num_features"] + artifact["cat_features"]
    X = features.reindex(columns=cols)
    cat = artifact["cat_features"]
    X[cat] = X[cat].astype(object).where(X[cat].notna(), np.nan)
    return pd.Series(artifact["model"].predict_proba(X)[:, 1], index=features.index)
