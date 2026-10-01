"""Build income features and write reports/income_{metrics,predictions}."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from creditforyou.evaluation import evaluate_income_dataset


if __name__ == "__main__":
    _, metrics = evaluate_income_dataset()
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
