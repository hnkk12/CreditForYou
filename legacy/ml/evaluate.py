"""
Model Evaluation Module for CreditForYou ML Pipeline.

Evaluates Logistic Regression (Primary Model)
on test data, computing ROC-AUC, Precision, Recall, F1-score, and Confusion Matrix.
Prints performance report table.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


# Model Evaluation
def evaluate_single_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> Dict[str, Any]:
    """
    Evaluates a single model on test dataset.
    """
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob >= 0.50).astype(int)

    roc_auc = roc_auc_score(y_test, y_pred_prob)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "model_name": model_name,
        "roc_auc": float(roc_auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist(),
        "y_pred_prob": y_pred_prob,
        "y_pred": y_pred
    }


def evaluate_models(train_results: Dict[str, Any] = None) -> Dict[str, Dict[str, Any]]:
    """
    Runs model evaluation for Logistic Regression model.
    """
    if train_results is None:
        from ml.train import train_models
        train_results = train_models()

    logistic_model = train_results["logistic_model"]
    X_test = train_results["X_test"]
    y_test = train_results["y_test"]

    res_logistic = evaluate_single_model(logistic_model, X_test, y_test, "Logistic Regression (Primary)")

    # Print clean performance report table
    print("\n" + "="*70)
    print("MODEL EVALUATION & PERFORMANCE REPORT")
    print("="*70)
    
    metrics_df = pd.DataFrame([
        {
            "Model": res_logistic["model_name"],
            "ROC-AUC": f"{res_logistic['roc_auc']:.4f}",
            "Precision": f"{res_logistic['precision']:.4f}",
            "Recall": f"{res_logistic['recall']:.4f}",
            "F1-Score": f"{res_logistic['f1_score']:.4f}",
            "Confusion Matrix (TN, FP, FN, TP)": str(res_logistic['confusion_matrix'])
        }
    ])
    
    print(metrics_df.to_string(index=False))
    print("="*70 + "\n")

    return {
        "logistic": res_logistic
    }


if __name__ == "__main__":
    eval_results = evaluate_models()

