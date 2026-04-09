from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train import TrainingConfig, train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HeartCycle models.")
    # parser.add_argument("--data-dir", type=Path, default=Path("outputs/datasets"))
    # parser.add_argument("--output-dir", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--data-dir", type=Path, default=ROOT / "outputs/datasets")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/runs")
    parser.add_argument("--model", choices=("cnn", "cnn_lstm"), default="cnn_lstm")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()

    config = TrainingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    report = train_model(config)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
