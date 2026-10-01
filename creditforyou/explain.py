"""Small model-agnostic local sensitivity explanation for the PD pipeline."""

import numpy as np
import pandas as pd

from .model import predict_pd


def explain_pd(artifact, features: pd.DataFrame, top_n=5) -> list[dict]:
    """Compare prediction with each feature masked; this is not SHAP/causality."""
    if artifact is None or features.empty:
        return []
    row = features.iloc[[0]].copy()
    base = float(predict_pd(artifact, row).iloc[0])
    explanations = []
    for feature in artifact["num_features"] + artifact["cat_features"]:
        masked = row.copy()
        masked[feature] = np.nan
        masked_pd = float(predict_pd(artifact, masked).iloc[0])
        delta = base - masked_pd
        explanations.append({
            "feature": feature,
            "value": None if pd.isna(row.iloc[0][feature]) else row.iloc[0][feature],
            "pd_delta_vs_missing": delta,
            "direction": "higher predicted risk" if delta > 0 else "lower predicted risk",
        })
    return sorted(explanations, key=lambda item: abs(item["pd_delta_vs_missing"]), reverse=True)[:top_n]
