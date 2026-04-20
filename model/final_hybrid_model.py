from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.cnn_improved import ImprovedCNN

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover
    XGBRegressor = None


def load_split(data_dir: Path, split_name: str) -> Dict[str, np.ndarray]:
    """
    Load one split from the clipped dataset.

    `y_reference` is preferred because it stores the clean evaluation target in
    [PEP, AVC] space. That keeps the final hybrid evaluation directly aligned
    with the project-wide metric reporting convention.
    """
    split = np.load(data_dir / f"{split_name}.npz", allow_pickle=True)
    return {
        "x": split["x"].astype(np.float32),
        "y_true": split["y_reference"].astype(np.float32) if "y_reference" in split.files else split["y"].astype(np.float32),
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    errors = y_pred - y_true
    mae = np.mean(np.abs(errors), axis=0)
    rmse = np.sqrt(np.mean(np.square(errors), axis=0))
    return {
        "pep_mae_ms": float(mae[0]),
        "avc_mae_ms": float(mae[1]),
        "pep_rmse_ms": float(rmse[0]),
        "avc_rmse_ms": float(rmse[1]),
        "mean_mae_ms": float(mae.mean()),
        "mean_rmse_ms": float(rmse.mean()),
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


class FinalHybridPredictor:
    """
    Final hybrid predictor:
    - CNN predicts PEP
    - XGBoost predicts AVC from CNN-derived features

    CRITICAL: Feature extraction and XGBoost must use SAME CNN
    Otherwise predictions will break
    """

    def __init__(
        self,
        *,
        cnn_weights_path: Path,
        cnn_report_path: Path,
        avc_xgb_model_path: Path,
        device: torch.device,
    ) -> None:
        self.device = device
        self.cnn_model = self._load_cnn_model(cnn_weights_path)
        self.pep_mean, self.pep_std = self._load_pep_normalization(cnn_report_path)
        self.avc_model = self._load_xgb_model(avc_xgb_model_path)

    def _load_cnn_model(self, weights_path: Path) -> ImprovedCNN:
        model = ImprovedCNN().to(self.device)
        state_dict = torch.load(weights_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def _load_pep_normalization(self, report_path: Path) -> tuple[float, float]:
        with report_path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        return float(report["target_mean_ms"][0]), float(report["target_std_ms"][0])

    def _load_xgb_model(self, model_path: Path):
        if XGBRegressor is None:
            raise ImportError(
                "xgboost is not installed in the current environment. "
                "Install it before running model/final_hybrid_model.py."
            )
        model = XGBRegressor()
        model.load_model(model_path)
        return model

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return CNN features for the XGBoost AVC regressor.

        This method is exposed explicitly so the final hybrid pipeline mirrors
        the project design: CNN for signal representation, XGBoost for AVC.
        """
        features = self.cnn_model.features(x)
        return features.view(features.size(0), -1)

    def predict_batch(
        self,
        x: np.ndarray,
        *,
        precomputed_features: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Predict [PEP, AVC] in milliseconds for one batch of inputs.

        PEP comes from the CNN regression head and is denormalized using the
        training statistics stored in the clipped CNN report. AVC comes from the
        XGBoost regressor using either precomputed features or on-the-fly CNN
        features if no saved features are provided.
        """
        x_tensor = torch.from_numpy(x).to(self.device)
        with torch.no_grad():
            cnn_output = self.cnn_model(x_tensor)
            pep_pred_norm = cnn_output[:, 0]
            pep_pred_ms = pep_pred_norm.cpu().numpy() * self.pep_std + self.pep_mean

            if precomputed_features is None:
                features_np = self.extract_features(x_tensor).cpu().numpy()
            else:
                features_np = precomputed_features

        avc_pred_ms = self.avc_model.predict(features_np)
        return np.stack([pep_pred_ms, avc_pred_ms], axis=1).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the final hybrid CNN (PEP) + XGBoost (AVC) pipeline.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "outputs" / "datasets" / "dataset_clipped")
    parser.add_argument("--feature-dir", type=Path, default=ROOT / "outputs" / "features")
    parser.add_argument("--cnn-weights", type=Path, default=ROOT / "outputs" / "runs" / "cnn_improved_v2_best_model.pt")
    parser.add_argument("--cnn-report", type=Path, default=ROOT / "outputs" / "runs" / "cnn_improved_v2_report.json")
    parser.add_argument("--avc-xgb-model", type=Path, default=ROOT / "outputs" / "runs" / "xgb_avc_model.json")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "outputs" / "runs")
    parser.add_argument("--plots-dir", type=Path, default=ROOT / "outputs" / "plots")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    args = parser.parse_args()

    args.runs_dir.mkdir(parents=True, exist_ok=True)
    args.plots_dir.mkdir(parents=True, exist_ok=True)

    split = load_split(args.data_dir, args.split)
    feature_path = args.feature_dir / f"{args.split}_X_features.npy"
    precomputed_features = np.load(feature_path) if feature_path.exists() else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = FinalHybridPredictor(
        cnn_weights_path=args.cnn_weights,
        cnn_report_path=args.cnn_report,
        avc_xgb_model_path=args.avc_xgb_model,
        device=device,
    )

    y_pred = predictor.predict_batch(split["x"], precomputed_features=precomputed_features)
    y_true = split["y_true"]
    metrics = compute_metrics(y_true, y_pred)

    report_path = args.runs_dir / "final_hybrid_report.json"
    prediction_plot_path = args.plots_dir / "final_hybrid_predictions.png"
    error_histogram_path = args.plots_dir / "final_hybrid_error_histogram.png"

    plot_predictions(y_true, y_pred, prediction_plot_path)
    plot_error_histogram(y_true, y_pred, error_histogram_path)

    report = {
        "split": args.split,
        "cnn_weights_path": str(args.cnn_weights),
        "cnn_report_path": str(args.cnn_report),
        "avc_xgb_model_path": str(args.avc_xgb_model),
        "feature_source": str(feature_path) if precomputed_features is not None else "on_the_fly_cnn_features",
        "test_metrics" if args.split == "test" else "metrics": metrics,
        "artifacts": {
            "report_path": str(report_path),
            "prediction_plot_path": str(prediction_plot_path),
            "error_histogram_path": str(error_histogram_path),
        },
    }
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
