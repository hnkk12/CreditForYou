"""Train the final PD baseline from resolved Lending Club outcomes."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from creditforyou.config import DEFAULT_MODEL_PATH, LENDING_CLUB_PATH, REPORTS_DIR
from creditforyou.model import train


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loans", type=Path, default=LENDING_CLUB_PATH)
    parser.add_argument("--max-rows", type=int, default=300_000)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()
    artifact = train(args.loans, feature_set="core", max_rows=args.max_rows, model_path=args.model)
    report = {k: v for k, v in artifact.items() if k != "model"}
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "credit_model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(args.model.with_suffix(".metrics.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
