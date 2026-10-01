"""Validate synthetic banking key and type integrity."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from creditforyou.datasets import validate_synthetic_joins


if __name__ == "__main__":
    print(json.dumps(validate_synthetic_joins(), ensure_ascii=False, indent=2))
