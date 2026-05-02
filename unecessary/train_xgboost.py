from __future__ import annotations


import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

try:
    from xgboost import XGBRegressor
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "xgboost is not installed in the virtual environment. "
        "Install it before running model/train_xgboost.py."
    ) from exc


def load_feature_split(feature_dir: Path, split_name: str) -> tuple[np.ndarray, np.ndarray]:
    X = np.load(feature_dir / f"{split_name}_X_features.npy")
    y = np.load(feature_dir / f"{split_name}_y_targets.npy")
    return X, y


def compute_metrics(y_true, y_pred):
    import numpy as np
    error = y_pred - y_true
    
    mae = np.mean(np.abs(error), axis=0)
    rmse = np.sqrt(np.mean(np.square(error), axis=0))
    
    ss_res = np.sum(error**2, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0))**2, axis=0)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    medae = np.median(np.abs(error), axis=0)
    max_err = np.max(np.abs(error), axis=0)
    bias = np.mean(error, axis=0)
    acc_10 = np.mean(np.abs(error) <= 10.0, axis=0) * 100.0
    acc_20 = np.mean(np.abs(error) <= 20.0, axis=0) * 100.0

    return {
        "avo_mae_ms": float(mae[0]), "avc_mae_ms": float(mae[1]), "mean_mae_ms": float(mae.mean()),
        "avo_rmse_ms": float(rmse[0]), "avc_rmse_ms": float(rmse[1]), "mean_rmse_ms": float(rmse.mean()),
        "avo_r2": float(r2[0]), "avc_r2": float(r2[1]), "mean_r2": float(r2.mean()),
        "avo_medae_ms": float(medae[0]), "avc_medae_ms": float(medae[1]), "mean_medae_ms": float(medae.mean()),
        "avo_max_err_ms": float(max_err[0]), "avc_max_err_ms": float(max_err[1]), "mean_max_err_ms": float(max_err.mean()),
        "avo_bias_ms": float(bias[0]), "avc_bias_ms": float(bias[1]), "mean_bias_ms": float(bias.mean()),
        "avo_acc_10ms_%": float(acc_10[0]), "avc_acc_10ms_%": float(acc_10[1]), "mean_acc_10ms_%": float(acc_10.mean()),
        "avo_acc_20ms_%": float(acc_20[0]), "avc_acc_20ms_%": float(acc_20[1]), "mean_acc_20ms_%": float(acc_20.mean()),
    }


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for index, target_name in enumerate(["PEP", "AVC"]):
        axes[index].scatter(y_true[:, index], y_pred[:, index], alpha=0.6, s=16)
        min_value = min(float(y_true[:, index].min()), float(y_pred[:, index].min()))
        max_value = max(float(y_true[:, index].max()), float(y_pred[:, index].max()))
        axes[index].plot([min_value, max_value], [min_value, max_value], "r--", linewidth=1.5)
        axes[index].set_title(f"{target_name}: Predicted vs True")
        axes[index].set_xlabel("True (ms)")
        axes[index].set_ylabel("Predicted (ms)")
        axes[index].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_error_histogram(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    errors = y_pred - y_true
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for index, target_name in enumerate(["PEP Error", "AVC Error"]):
        axes[index].hist(errors[:, index], bins=30, alpha=0.8, edgecolor="black")
        axes[index].axvline(0.0, color="red", linestyle="--", linewidth=1.2)
        axes[index].set_title(target_name)
        axes[index].set_xlabel("Prediction Error (ms)")
        axes[index].set_ylabel("Count")
        axes[index].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_regressor() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
    )


def main() -> None:
    # IMPORTANT: XGBoost must be trained on features from cnn_improved_v2
    # Do NOT reuse old XGBoost models
    parser = argparse.ArgumentParser(description="Train XGBoost regressors on CNN-extracted features.")
    parser.add_argument("--feature-dir", type=Path, default=Path("outputs/features"))
    parser.add_argument("--runs-dir", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--plots-dir", type=Path, default=Path("outputs/plots"))
    args = parser.parse_args()

    args.runs_dir.mkdir(parents=True, exist_ok=True)
    args.plots_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train = load_feature_split(args.feature_dir, "train")
    X_val, y_val = load_feature_split(args.feature_dir, "val")
    X_test, y_test = load_feature_split(args.feature_dir, "test")

    model_pep = build_regressor()
    model_avc = build_regressor()

    # Validation is kept separate and provided through early-stopping style eval
    # sets so the train/val/test split remains leakage-safe.
    model_pep.fit(X_train, y_train[:, 0], eval_set=[(X_val, y_val[:, 0])], verbose=False)
    model_avc.fit(X_train, y_train[:, 1], eval_set=[(X_val, y_val[:, 1])], verbose=False)

    pep_pred = model_pep.predict(X_test)
    avc_pred = model_avc.predict(X_test)
    y_pred = np.stack([pep_pred, avc_pred], axis=1).astype(np.float32)

    metrics = compute_metrics(y_test, y_pred)

    pep_model_path = args.runs_dir / "xgb_pep_model.json"
    avc_model_path = args.runs_dir / "xgb_avc_model.json"
    report_path = args.runs_dir / "xgb_report.json"
    pred_plot_path = args.plots_dir / "xgb_predictions.png"
    error_plot_path = args.plots_dir / "xgb_error_histogram.png"

    model_pep.save_model(pep_model_path)
    model_avc.save_model(avc_model_path)
    plot_predictions(y_test, y_pred, pred_plot_path)
    plot_error_histogram(y_test, y_pred, error_plot_path)

    report = {
        "feature_dir": str(args.feature_dir),
        "train_shape": list(X_train.shape),
        "val_shape": list(X_val.shape),
        "test_shape": list(X_test.shape),
        "test_metrics": metrics,
        "artifacts": {
            "pep_model_path": str(pep_model_path),
            "avc_model_path": str(avc_model_path),
            "report_path": str(report_path),
            "prediction_plot_path": str(pred_plot_path),
            "error_histogram_path": str(error_plot_path),
        },
    }
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
