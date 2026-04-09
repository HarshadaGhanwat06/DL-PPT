from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import DatasetConfig, inspect_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the HeartCycle dataset.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()

    config = DatasetConfig(data_dir=args.data_dir)
    summary = inspect_dataset(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
