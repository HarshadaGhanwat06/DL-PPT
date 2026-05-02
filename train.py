from __future__ import annotations

import os
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

from models import (
    CNNLSTMRegressor,
    CNNRegressor,
    ResNet1DRegressor,
    TCNRegressor,
    TransformerRegressor,
    require_torch,
)

require_torch()
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainingConfig:
    data_dir: Path = Path("outputs/datasets")
    output_dir: Path = Path("outputs/runs")
    plots_dir: Path = Path("outputs/plots")
    model_name: str = "cnn_lstm"
    seed: int = 42
    epochs: int = 25
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 5
    # Physiological consistency loss weight (λ).
    # At λ=0 this reduces to plain SmoothL1.
    consistency_lambda: float = 0.1
    # Data augmentation: set False to disable for ablation studies
    augment: bool = True


def _ensure_runtime_dirs(base_dir: Path) -> Path:
    runtime_dir = base_dir / ".runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TMP", str(runtime_dir))
    os.environ.setdefault("TEMP", str(runtime_dir))
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(runtime_dir / "torchinductor"))
    return runtime_dir


def _config_to_jsonable_dict(config: TrainingConfig) -> Dict[str, object]:
    raw = asdict(config)
    return {key: str(value) if isinstance(value, Path) else value for key, value in raw.items()}


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_split(data_dir: Path, split_name: str) -> Dict[str, np.ndarray]:
    raw = np.load(data_dir / f"{split_name}.npz", allow_pickle=True)
    return {key: raw[key] for key in raw.files}


def _ensure_plot_output_paths(config: TrainingConfig) -> Dict[str, Path]:
    """
    Create a per-model plot folder and define all output artifact paths.

    Repeated runs intentionally reuse the same filenames so comparison stays
    simple. When files already exist, the trainer prints that they will be
    overwritten instead of silently replacing them.
    """
    plot_subdir = config.plots_dir / config.model_name
    plot_subdir.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "model_path": config.output_dir / f"{config.model_name}.pt",
        "report_path": config.output_dir / f"{config.model_name}_report.json",
        "loss_curve_path": plot_subdir / f"{config.model_name}_loss_curve.png",
        "val_mae_curve_path": plot_subdir / f"{config.model_name}_val_mae_curve.png",
        "predicted_vs_true_path": plot_subdir / f"{config.model_name}_predicted_vs_true.png",
        "error_histogram_path": plot_subdir / f"{config.model_name}_error_histogram.png",
    }

    existing = [str(path) for path in output_paths.values() if path.exists()]
    print(f"[DEBUG] Plot output folder: {plot_subdir}")
    if existing:
        print("[DEBUG] Existing artifacts detected. This run will overwrite them:")
        for path in existing:
            print(f"[DEBUG]   {path}")
    else:
        print("[DEBUG] No existing artifacts found. New files will be created.")

    return output_paths


def _make_loader(
    split_data: Dict[str, np.ndarray],
    batch_size: int,
    shuffle: bool,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> DataLoader:
    # Use only the first channel (dzdt) as these models expect 1 channel
    x_data = split_data["x"][:, 0, :] if split_data["x"].ndim == 3 else split_data["x"]
    x = torch.tensor(x_data, dtype=torch.float32).unsqueeze(1)
    y = torch.tensor((split_data["y"] - target_mean) / target_std, dtype=torch.float32)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)


def augment_batch(x: torch.Tensor) -> torch.Tensor:
    """Apply randomised 1D signal augmentations during training only.

    Four independent transforms are composed per batch:
    1. Gaussian jitter  – additive noise (σ ≈ 2 % of signal std per sample)
    2. Amplitude scale  – multiplicative factor in [0.9, 1.1]
    3. Time warp        – random stretch/compress then resample to original length
    4. R-peak offset    – circular roll by ±5 samples

    All operations are differentiable where possible and stay on the same device
    as the input tensor.
    """
    device = x.device
    B, C, L = x.shape

    # 1. Gaussian jitter: σ = 2% of each sample's std, clipped to a small range
    signal_std = x.std(dim=-1, keepdim=True).clamp(min=1e-6)
    noise = torch.randn_like(x) * (0.02 * signal_std)
    x = x + noise

    # 2. Amplitude scaling: per-sample scalar in [0.9, 1.1]
    scale = (torch.rand(B, 1, 1, device=device) * 0.2 + 0.9)
    x = x * scale

    # 3. Time warping: stretch by factor in [0.95, 1.05], then resample to L
    warp = (torch.rand(1, device=device) * 0.10 + 0.95).item()  # scalar
    warped_len = max(1, int(round(L * warp)))
    x = F.interpolate(x, size=warped_len, mode="linear", align_corners=False)
    x = F.interpolate(x, size=L, mode="linear", align_corners=False)

    # 4. R-peak offset: circular shift by a random integer in [-5, 5]
    shift = int(torch.randint(-5, 6, (1,)).item())
    if shift != 0:
        x = torch.roll(x, shifts=shift, dims=-1)

    return x


class PhysiologicalConsistencyLoss(nn.Module):
    """SmoothL1 loss augmented with a soft physiological constraint.

    The constraint encodes two known relationships between the four cardiac
    timing events (in normalised space):

        LVET ≈ AVC − AVO
        PEP  ≈ AVO

    Because the dataset here contains only [AVO, AVC] (indices 0 and 1) as
    training targets, the consistency term checks only the constraint that
    AVC > AVO (AVC − AVO > 0), i.e. the valve closes after it opens.  When
    targets include a third LVET column the full |LVET − (AVC − AVO)| penalty
    is automatically activated.

    Parameters
    ----------
    lambda_consistency:
        Weight of the constraint term relative to the SmoothL1 term.
        Typical range: 0.05–0.2.  Set to 0 to fall back to plain SmoothL1.
    """

    def __init__(self, lambda_consistency: float = 0.1) -> None:
        super().__init__()
        self.lambda_c = lambda_consistency
        self.smooth_l1 = nn.SmoothL1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        base_loss = self.smooth_l1(pred, target)
        if self.lambda_c == 0.0:
            return base_loss

        consistency_terms = []

        # Constraint: AVC prediction (index 1) must exceed AVO prediction (index 0)
        # Penalise only when pred_AVC < pred_AVO (soft ReLU hinge)
        if pred.shape[-1] >= 2:
            diff_pred = pred[:, 1] - pred[:, 0]   # AVC − AVO
            diff_true = target[:, 1] - target[:, 0]
            # Both predictions and targets should satisfy AVC > AVO
            consistency_terms.append(F.relu(-diff_pred).mean())
            consistency_terms.append(F.mse_loss(diff_pred, diff_true))

        # Constraint: LVET ≈ AVC − AVO when a third target column is present
        if pred.shape[-1] >= 3:
            lvet_pred = pred[:, 2]
            lvet_implied = pred[:, 1] - pred[:, 0]
            consistency_terms.append(F.smooth_l1_loss(lvet_pred, lvet_implied))

        if not consistency_terms:
            return base_loss

        consistency_loss = torch.stack(consistency_terms).mean()
        return base_loss + self.lambda_c * consistency_loss


def _build_model(model_name: str, input_length: int):
    if model_name == "cnn":
        return CNNRegressor(input_length=input_length)
    if model_name == "cnn_lstm":
        return CNNLSTMRegressor()
    if model_name == "resnet":
        return ResNet1DRegressor()
    if model_name == "tcn":
        return TCNRegressor()
    if model_name == "transformer":
        return TransformerRegressor()
    raise ValueError(f"Unsupported model_name={model_name!r}")


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    error = y_pred - y_true
    
    # 1. MAE
    mae = np.mean(np.abs(error), axis=0)
    
    # 2. RMSE
    rmse = np.sqrt(np.mean(np.square(error), axis=0))
    
    # 3. R² Score
    ss_res = np.sum(error**2, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0))**2, axis=0)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    # 4. Median Absolute Error
    medae = np.median(np.abs(error), axis=0)
    
    # 5. Max Error
    max_err = np.max(np.abs(error), axis=0)
    
    # 6. Bias (Mean Error)
    bias = np.mean(error, axis=0)
    
    # 7. Within ±10 ms Accuracy
    acc_10 = np.mean(np.abs(error) <= 10.0, axis=0) * 100.0
    
    # 8. Within ±20 ms Accuracy
    acc_20 = np.mean(np.abs(error) <= 20.0, axis=0) * 100.0

    return {
        "avo_mae_ms": float(mae[0]),
        "avc_mae_ms": float(mae[1]),
        "mean_mae_ms": float(mae.mean()),
        
        "avo_rmse_ms": float(rmse[0]),
        "avc_rmse_ms": float(rmse[1]),
        "mean_rmse_ms": float(rmse.mean()),
        
        "avo_r2": float(r2[0]),
        "avc_r2": float(r2[1]),
        "mean_r2": float(r2.mean()),
        
        "avo_medae_ms": float(medae[0]),
        "avc_medae_ms": float(medae[1]),
        "mean_medae_ms": float(medae.mean()),
        
        "avo_max_err_ms": float(max_err[0]),
        "avc_max_err_ms": float(max_err[1]),
        "mean_max_err_ms": float(max_err.mean()),
        
        "avo_bias_ms": float(bias[0]),
        "avc_bias_ms": float(bias[1]),
        "mean_bias_ms": float(bias.mean()),
        
        "avo_acc_10ms_%": float(acc_10[0]),
        "avc_acc_10ms_%": float(acc_10[1]),
        "mean_acc_10ms_%": float(acc_10.mean()),
        
        "avo_acc_20ms_%": float(acc_20[0]),
        "avc_acc_20ms_%": float(acc_20[1]),
        "mean_acc_20ms_%": float(acc_20.mean()),
    }


def _evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> Dict[str, float]:
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            outputs = model(x_batch).cpu().numpy()
            predictions.append(outputs)
            targets.append(y_batch.numpy())
    y_true = np.concatenate(targets, axis=0) * target_std + target_mean
    y_pred = np.concatenate(predictions, axis=0) * target_std + target_mean
    return _compute_metrics(y_true, y_pred)


def _evaluate_model_with_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> Dict[str, object]:
    """
    Return both scalar metrics and denormalized predictions for plotting.
    """
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            outputs = model(x_batch).cpu().numpy()
            predictions.append(outputs)
            targets.append(y_batch.numpy())
    y_true = np.concatenate(targets, axis=0) * target_std + target_mean
    y_pred = np.concatenate(predictions, axis=0) * target_std + target_mean
    return {
        "metrics": _compute_metrics(y_true, y_pred),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def _plot_results(
    history: list[Dict[str, float]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_paths: Dict[str, Path],
    model_name: str,
) -> Dict[str, str]:
    """
    Save the standard set of training/evaluation plots for a model.
    """
    plt.figure(figsize=(8, 5))
    plt.plot([epoch["train_loss"] for epoch in history], linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title(f"{model_name.upper()} Training Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_paths["loss_curve_path"], dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot([epoch["mean_mae_ms"] for epoch in history], label="Mean Val MAE", linewidth=2)
    plt.plot([epoch["avo_mae_ms"] for epoch in history], label="PEP Val MAE", linewidth=1.6)
    plt.plot([epoch["avc_mae_ms"] for epoch in history], label="AVC Val MAE", linewidth=1.6)
    plt.xlabel("Epoch")
    plt.ylabel("Validation MAE (ms)")
    plt.title(f"{model_name.upper()} Validation MAE")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_paths["val_mae_curve_path"], dpi=200)
    plt.close()

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
    fig.savefig(output_paths["predicted_vs_true_path"], dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    errors = y_pred - y_true
    for index, target_name in enumerate(["PEP Error", "AVC Error"]):
        axes[index].hist(errors[:, index], bins=30, alpha=0.8, edgecolor="black")
        axes[index].axvline(0.0, color="red", linestyle="--", linewidth=1.2)
        axes[index].set_title(target_name)
        axes[index].set_xlabel("Prediction Error (ms)")
        axes[index].set_ylabel("Count")
        axes[index].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_paths["error_histogram_path"], dpi=200)
    plt.close(fig)

    return {key: str(value) for key, value in output_paths.items() if key.endswith("_path")}


def _baseline_metrics(split_data: Dict[str, np.ndarray]) -> Dict[str, float]:
    return _compute_metrics(split_data["y"], split_data["baseline"])


def train_model(config: TrainingConfig) -> Dict[str, object]:
    _set_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.plots_dir.mkdir(parents=True, exist_ok=True)
    _ensure_runtime_dirs(config.output_dir.parent)
    output_paths = _ensure_plot_output_paths(config)

    train_data = _load_split(config.data_dir, "train")
    val_data = _load_split(config.data_dir, "val")
    test_data = _load_split(config.data_dir, "test")

    target_mean = train_data["y"].mean(axis=0).astype(np.float32)
    target_std = train_data["y"].std(axis=0).astype(np.float32) + 1e-6

    train_loader = _make_loader(train_data, config.batch_size, shuffle=True, target_mean=target_mean, target_std=target_std)
    val_loader = _make_loader(val_data, config.batch_size, shuffle=False, target_mean=target_mean, target_std=target_std)
    test_loader = _make_loader(test_data, config.batch_size, shuffle=False, target_mean=target_mean, target_std=target_std)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(config.model_name, input_length=train_data["x"].shape[1]).to(device)
    
    # EMA Model setup
    ema_model = _build_model(config.model_name, input_length=train_data["x"].shape[1]).to(device)
    ema_model.load_state_dict(model.state_dict())
    ema_decay = 0.99
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    
    # Cosine LR schedule with warmup
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        epochs=config.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=5.0 / config.epochs if config.epochs > 5 else 0.3, # ~5 epochs warmup
    )
    
    criterion = PhysiologicalConsistencyLoss(lambda_consistency=config.consistency_lambda)

    best_state = None
    best_val_mae = float("inf")
    patience_counter = 0
    history = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            # Apply augmentation during training only
            if config.augment:
                x_batch = augment_batch(x_batch)
            optimizer.zero_grad()
            predictions = model(x_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            
            # Gradient clipping at 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            # EMA Update
            with torch.no_grad():
                for param, ema_param in zip(model.parameters(), ema_model.parameters()):
                    ema_param.data.mul_(ema_decay).add_(param.data, alpha=1 - ema_decay)
            
            running_loss += float(loss.item()) * x_batch.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        
        # Evaluate using EMA weights
        val_metrics = _evaluate_model(ema_model, val_loader, device, target_mean=target_mean, target_std=target_std)
        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss,
            **val_metrics,
        }
        history.append(epoch_result)

        if val_metrics["mean_mae_ms"] < best_val_mae:
            best_val_mae = val_metrics["mean_mae_ms"]
            best_state = {key: value.cpu() for key, value in ema_model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation uses the best EMA weights loaded into model
    train_metrics = _evaluate_model(model, train_loader, device, target_mean=target_mean, target_std=target_std)
    val_metrics = _evaluate_model(model, val_loader, device, target_mean=target_mean, target_std=target_std)
    test_result = _evaluate_model_with_predictions(model, test_loader, device, target_mean=target_mean, target_std=target_std)
    test_metrics = test_result["metrics"]
    test_baseline = _baseline_metrics(test_data)
    plot_paths = _plot_results(history, test_result["y_true"], test_result["y_pred"], output_paths, config.model_name)

    torch.save(model.state_dict(), output_paths["model_path"])

    report: Dict[str, object] = {
        "config": _config_to_jsonable_dict(config),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "baseline_test_metrics": test_baseline,
        "target_mean_ms": target_mean.tolist(),
        "target_std_ms": target_std.tolist(),
        "history": history,
        "artifacts": {
            "model_path": str(output_paths["model_path"]),
            "report_path": str(output_paths["report_path"]),
            **plot_paths,
        },
    }
    with output_paths["report_path"].open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train cardiac timing prediction models")
    parser.add_argument("--model", type=str, default="cnn_lstm",
                        choices=["cnn", "cnn_lstm", "resnet", "tcn", "transformer"],
                        help="Model architecture to train")
    parser.add_argument("--epochs", type=int, default=25,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate for optimizer")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay (L2 regularization)")
    parser.add_argument("--consistency_lambda", type=float, default=0.1,
                        help="Physiological consistency loss weight")
    parser.add_argument("--no_augment", action="store_true",
                        help="Disable data augmentation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        weight_decay=args.weight_decay,
        consistency_lambda=args.consistency_lambda,
        augment=not args.no_augment,
        seed=args.seed,
    )
    
    print(f"\n{'='*70}")
    print(f"Training {config.model_name.upper()}")
    print(f"{'='*70}")
    print(f"Epochs: {config.epochs} | Batch Size: {config.batch_size}")
    print(f"Learning Rate: {config.learning_rate} | Patience: {config.patience}")
    print(f"Data Augmentation: {config.augment}")
    print(f"{'='*70}\n")
    
    result = train_model(config)
    
    # Print final results
    test_mae = result["test_metrics"]["mean_mae_ms"]
    test_rmse = result["test_metrics"]["mean_rmse_ms"]
    avo_mae = result["test_metrics"]["avo_mae_ms"]
    avc_mae = result["test_metrics"]["avc_mae_ms"]
    epochs_trained = len(result["history"])
    
    print(f"\n{'='*70}")
    print(f"✅ Training Complete: {config.model_name.upper()}")
    print(f"{'='*70}")
    print(f"Epochs Trained:   {epochs_trained}")
    print(f"Test MAE:         {test_mae:.2f} ms")
    print(f"Test RMSE:        {test_rmse:.2f} ms")
    print(f"AVO MAE:          {avo_mae:.2f} ms")
    print(f"AVC MAE:          {avc_mae:.2f} ms")
    print(f"Model saved to:   outputs/runs/{config.model_name}.pt")
    print(f"Report saved to:  outputs/runs/{config.model_name}_report.json")
    print(f"{'='*70}\n")
