"""
CreditForYou ML / Credit-Scoring Pipeline Package.
"""

from .rule_engine import compute_rule_based_credit_score, run_what_if_simulation
from .data_loader import load_combined_dataset
from .target_generation import generate_target_label
from .feature_engineering import engineer_features
from .preprocessing import PipelinePreprocessor
from .feature_selection import perform_feature_selection, load_selected_features
from .train import train_models
from .evaluate import evaluate_models
from .predict import predict_credit_risk, load_inference_artifacts
from .explain import explain_applicant_risk, generate_global_feature_importance

__all__ = [
    "compute_rule_based_credit_score",
    "run_what_if_simulation",
    "load_combined_dataset",
    "generate_target_label",
    "engineer_features",
    "PipelinePreprocessor",
    "perform_feature_selection",
    "load_selected_features",
    "train_models",
    "evaluate_models",
    "predict_credit_risk",
    "load_inference_artifacts",
    "explain_applicant_risk",
    "generate_global_feature_importance"
]
