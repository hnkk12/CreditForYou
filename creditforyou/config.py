"""Đường dẫn dự án và nạp cấu hình (config/*.json)."""

import json
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = DATA_DIR / "sample"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Dataset folder supplied with this repository.  Keeping this lookup in one
# place avoids baking the folder name into the income/model modules.
ISB_DIR = PROJECT_ROOT / "dataset isb"
SYNTHETIC_DATA_DIR = ISB_DIR / "data"
LENDING_CLUB_PATH = ISB_DIR / "accepted_2007_to_2018Q4.csv"

DEFAULT_MODEL_PATH = MODELS_DIR / "pd_model.joblib"


@lru_cache(maxsize=None)
def _load_json(name: str) -> dict:
    with open(CONFIG_DIR / name, "r", encoding="utf-8") as f:
        return json.load(f)


def scoring_config() -> dict:
    return _load_json("scoring.json")


def schema_config() -> dict:
    return _load_json("schema.json")
