from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import DatasetConfig, build_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build beat-level datasets from HeartCycle HDF5 files.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/datasets")
    parser.add_argument("--target-length", type=int, default=160)
    parser.add_argument("--pre-r-ms", type=float, default=250.0)
    parser.add_argument("--post-r-ms", type=float, default=500.0)
    args = parser.parse_args()

    config = DatasetConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        target_length=args.target_length,
        pre_r_ms=args.pre_r_ms,
        post_r_ms=args.post_r_ms,
    )
    summary = build_dataset(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
