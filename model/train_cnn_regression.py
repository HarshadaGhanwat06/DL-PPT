from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'model'))

from cnn_regression import CNNRegressionConfig, train_cnn_regressor


def main() -> None:
    parser = argparse.ArgumentParser(description='Train a 1D CNN regressor on heartbeat segments.')
    parser.add_argument('--train-path', type=Path, default=ROOT / 'outputs/datasets/train.npz')
    parser.add_argument('--val-path', type=Path, default=ROOT / 'outputs/datasets/val.npz')
    parser.add_argument('--test-path', type=Path, default=ROOT / 'outputs/datasets/test.npz')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'model_ouput')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    config = CNNRegressionConfig(
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        patience=args.patience,
        seed=args.seed,
    )
    result = train_cnn_regressor(config)
    metrics = result['metrics']

    print('\nTest Metrics')
    print(f"PEP MAE:  {metrics['pep_mae']:.4f}")
    print(f"AVC MAE:  {metrics['avc_mae']:.4f}")
    print(f"PEP RMSE: {metrics['pep_rmse']:.4f}")
    print(f"AVC RMSE: {metrics['avc_rmse']:.4f}")
    print(f"Mean MAE: {metrics['mean_mae']:.4f}")
    print(f"Mean RMSE:{metrics['mean_rmse']:.4f}")

    print(f"\nArtifacts saved in: {result['output_dir']}")
    print(f"- Best model: {result['best_model_path']}")
    print(f"- Loss curve: {Path(result['output_dir']) / 'loss_curve.png'}")
    print(f"- Prediction plot: {Path(result['output_dir']) / 'predicted_vs_true.png'}")
    print(f"- Error histogram: {Path(result['output_dir']) / 'error_histogram.png'}")
    print(f"- Report: {Path(result['output_dir']) / 'report.json'}")


if __name__ == '__main__':
    main()
