"""Cache transaction/income features without modifying the raw dataset."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from creditforyou.config import PROCESSED_DIR
from creditforyou.datasets import synthetic_paths
from creditforyou.pipeline import run_income


if __name__ == "__main__":
    features, warnings = run_income(synthetic_paths()["transactions"])
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / "transaction_features.csv"
    features.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Saved {len(features)} customers to {path}")
    for warning in warnings:
        print(f"WARNING: {warning}")
