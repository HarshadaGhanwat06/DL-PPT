from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.cnn_dual_advanced import DualBranchAdvancedCNN

W_PEP = 0.4
W_LVET = 0.6


@dataclass
class DualBranchAdvancedConfig:
    data_dir: Path = Path("outputs/datasets")
    runs_dir: Path = Path("outputs/runs")
    plots_dir: Path = Path("outputs/plots")
    batch_size: int = 32
    epochs: int = 60
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 12
    seed: int = 42


def ensure_runtime_dirs(base_dir: Path) -> None:
    runtime_dir = base_dir / ".runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TMP", str(runtime_dir))
    os.environ.setdefault("TEMP", str(runtime_dir))
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(runtime_dir / "torchinductor"))


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(data_dir: Path) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load the ECG-enabled dataset and convert targets from [PEP, AVC] to
    [PEP, LVET], where LVET = AVC - PEP.
    """
    data_dir = Path(data_dir)
    splits: Dict[str, Dict[str, np.ndarray]] = {}
    for split_name in ("train", "val", "test"):
        split = np.load(data_dir / f"{split_name}.npz", allow_pickle=True)
        x = split["x"].astype(np.float32)
        if x.ndim != 3 or x.shape[1] != 2:
            raise ValueError(
                f"Expected {split_name}.npz x to have shape (N, 2, 160), got {x.shape!r}. "
                "Regenerate the ECG-enabled dataset with scripts/prepare_dataset.py."
            )

        y_avo_avc = split["y"].astype(np.float32)
        pep = y_avo_avc[:, 0]
        avc = y_avo_avc[:, 1]
        lvet = avc - pep
        y_pep_lvet = np.stack([pep, lvet], axis=1).astype(np.float32)

        splits[split_name] = {
            "x": x,
            "y": y_pep_lvet,
            "y_original": y_avo_avc,
        }
    return splits


def normalize_targets(
    train_y: np.ndarray,
    val_y: np.ndarray,
    test_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y_mean = train_y.mean(axis=0).astype(np.float32)
    y_std = (train_y.std(axis=0) + 1e-6).astype(np.float32)
    train_y_norm = (train_y - y_mean) / y_std
    val_y_norm = (val_y - y_mean) / y_std
    test_y_norm = (test_y - y_mean) / y_std
    return train_y_norm, val_y_norm, test_y_norm, y_mean, y_std


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    x_tensor = torch.from_numpy(x)
    y_tensor = torch.from_numpy(y)
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def build_model() -> DualBranchAdvancedCNN:
    return DualBranchAdvancedCNN()


def log_cosh_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable log-cosh loss.

    log(cosh(x)) is approximated in a stable form to avoid overflow for larger
    residuals while still behaving like a smooth robust regression loss.
    """
    error = pred - target
    return torch.mean(error + nn.functional.softplus(-2.0 * error) - math.log(2.0))


def weighted_log_cosh_loss(predictions: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    loss_pep = log_cosh_loss(predictions[:, 0], targets[:, 0])
    loss_lvet = log_cosh_loss(predictions[:, 1], targets[:, 1])
    loss = W_PEP * loss_pep + W_LVET * loss_lvet
    return loss, float(loss_pep.item()), float(loss_lvet.item())


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        predictions = model(inputs)
        loss, _, _ = weighted_log_cosh_loss(predictions, targets)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item()) * inputs.size(0)

    return running_loss / len(loader.dataset)


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


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    y_mean: np.ndarray,
    y_std: np.ndarray,
) -> Dict[str, object]:
    """
    Evaluate in AVC space even though the model predicts [PEP, LVET].

    Predictions are first denormalized back to [PEP, LVET]. AVC is then
    reconstructed as PEP + LVET so reported metrics remain directly comparable
    with the rest of the project.
    """
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for inputs, batch_targets in loader:
            inputs = inputs.to(device)
            outputs = model(inputs).cpu().numpy()
            predictions.append(outputs)
            targets.append(batch_targets.numpy())

    y_pred_norm = np.concatenate(predictions, axis=0)
    y_true_norm = np.concatenate(targets, axis=0)

    y_pred_pep_lvet = y_pred_norm * y_std + y_mean
    y_true_pep_lvet = y_true_norm * y_std + y_mean

    y_pred_avc_space = np.stack(
        [y_pred_pep_lvet[:, 0], y_pred_pep_lvet[:, 0] + y_pred_pep_lvet[:, 1]],
        axis=1,
    )
    y_true_avc_space = np.stack(
        [y_true_pep_lvet[:, 0], y_true_pep_lvet[:, 0] + y_true_pep_lvet[:, 1]],
        axis=1,
    )

    metrics = compute_metrics(y_true_avc_space, y_pred_avc_space)
    return {
        "metrics": metrics,
        "y_true": y_true_avc_space,
        "y_pred": y_pred_avc_space,
        "y_true_pep_lvet": y_true_pep_lvet,
        "y_pred_pep_lvet": y_pred_pep_lvet,
    }


def plot_results(
    history: Dict[str, list[float]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_paths: Dict[str, Path],
) -> Dict[str, str]:
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Weighted Log-Cosh Loss")
    plt.title("Dual-Branch Advanced CNN Training Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_paths["loss_curve_path"], dpi=200)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    target_names = ["PEP", "AVC"]
    for index, axis in enumerate(axes):
        axis.scatter(y_true[:, index], y_pred[:, index], alpha=0.6, s=16)
        min_value = min(float(y_true[:, index].min()), float(y_pred[:, index].min()))
        max_value = max(float(y_true[:, index].max()), float(y_pred[:, index].max()))
        axis.plot([min_value, max_value], [min_value, max_value], "r--", linewidth=1.5)
        axis.set_title(f"{target_names[index]}: Predicted vs True")
        axis.set_xlabel("True (ms)")
        axis.set_ylabel("Predicted (ms)")
        axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_paths["predicted_vs_true_path"], dpi=200)
    plt.close(fig)

    return {key: str(value) for key, value in output_paths.items() if key.endswith("_path")}


def ensure_output_paths_available(config: DualBranchAdvancedConfig) -> Dict[str, Path]:
    output_paths = {
        "best_model_path": config.runs_dir / "cnn_dual_advanced_best_model.pt",
        "report_path": config.runs_dir / "cnn_dual_advanced_report.json",
        "loss_curve_path": config.plots_dir / "cnn_dual_advanced_loss_curve.png",
        "predicted_vs_true_path": config.plots_dir / "cnn_dual_advanced_predicted_vs_true.png",
    }
    existing = [str(path) for path in output_paths.values() if path.exists()]
    return output_paths


def train_and_evaluate(config: DualBranchAdvancedConfig) -> Dict[str, object]:
    config.runs_dir.mkdir(parents=True, exist_ok=True)
    config.plots_dir.mkdir(parents=True, exist_ok=True)
    ensure_runtime_dirs(config.runs_dir.parent)
    set_seed(config.seed)
    output_paths = ensure_output_paths_available(config)

    data = load_data(config.data_dir)
    train_y_norm, val_y_norm, test_y_norm, y_mean, y_std = normalize_targets(
        data["train"]["y"],
        data["val"]["y"],
        data["test"]["y"],
    )

    train_loader = make_loader(data["train"]["x"], train_y_norm, batch_size=config.batch_size, shuffle=True)
    val_loader = make_loader(data["val"]["x"], val_y_norm, batch_size=config.batch_size, shuffle=False)
    test_loader = make_loader(data["test"]["x"], test_y_norm, batch_size=config.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    best_state = None
    best_val_mae = float("inf")
    epochs_without_improvement = 0
    history: Dict[str, list[float]] = {
        "train_loss": [],
        "val_mae_ms": [],
        "val_avo_mae_ms": [],
        "val_avc_mae_ms": [],
    }

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_result = evaluate(model, val_loader, device, y_mean, y_std)
        val_mae = float(val_result["metrics"]["mean_mae_ms"])
        val_avo_mae = float(val_result["metrics"]["avo_mae_ms"])
        val_avc_mae = float(val_result["metrics"]["avc_mae_ms"])

        history["train_loss"].append(train_loss)
        history["val_mae_ms"].append(val_mae)
        history["val_avo_mae_ms"].append(val_avo_mae)
        history["val_avc_mae_ms"].append(val_avc_mae)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save(best_state, output_paths["best_model_path"])
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        scheduler.step(val_mae)

        print(
            f"Epoch {epoch:02d}/{config.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val MAE: {val_mae:.4f} ms | "
            f"AVO MAE: {val_avo_mae:.4f} ms | "
            f"AVC MAE: {val_avc_mae:.4f} ms | "
            f"Best MAE: {best_val_mae:.4f} ms"
        )

        if epochs_without_improvement >= config.patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    if best_state is None:
        raise RuntimeError("Training finished without producing a best model checkpoint.")

    model.load_state_dict(best_state)
    train_result = evaluate(model, train_loader, device, y_mean, y_std)
    val_result = evaluate(model, val_loader, device, y_mean, y_std)
    test_result = evaluate(model, test_loader, device, y_mean, y_std)
    plot_paths = plot_results(history, test_result["y_true"], test_result["y_pred"], output_paths)

    report = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "target_definition": ["pep_ms", "lvet_ms"],
        "reconstructed_evaluation_targets": ["pep_ms", "avc_ms"],
        "target_mean_ms": y_mean.tolist(),
        "target_std_ms": y_std.tolist(),
        "input_channels": ["dzdt", "ecg"],
        "history": [
            {
                "epoch": epoch_index + 1,
                "train_loss": history["train_loss"][epoch_index],
                "val_mae_ms": history["val_mae_ms"][epoch_index],
                "val_avo_mae_ms": history["val_avo_mae_ms"][epoch_index],
                "val_avc_mae_ms": history["val_avc_mae_ms"][epoch_index],
            }
            for epoch_index in range(len(history["train_loss"]))
        ],
        "train_metrics": train_result["metrics"],
        "val_metrics": val_result["metrics"],
        "test_metrics": test_result["metrics"],
        "artifacts": {
            "best_model_path": str(output_paths["best_model_path"]),
            "report_path": str(output_paths["report_path"]),
            **plot_paths,
        },
        "loss_weights": {
            "pep": W_PEP,
            "lvet": W_LVET,
        },
    }

    with output_paths["report_path"].open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the advanced dual-branch CNN regressor for PEP and AVC.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "outputs" / "datasets")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "outputs" / "runs")
    parser.add_argument("--plots-dir", type=Path, default=ROOT / "outputs" / "plots")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = DualBranchAdvancedConfig(
        data_dir=args.data_dir,
        runs_dir=args.runs_dir,
        plots_dir=args.plots_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
    )
    report = train_and_evaluate(config)
    test_metrics = report["test_metrics"]

    print("="*45)
    print(" TEST METRICS ")
    print("="*45)
    print(f"[MAE]  AVO: {metrics['avo_mae_ms']:6.2f} ms | AVC: {metrics['avc_mae_ms']:6.2f} ms | Mean: {metrics['mean_mae_ms']:6.2f} ms")
    print(f"[RMSE] AVO: {metrics['avo_rmse_ms']:6.2f} ms | AVC: {metrics['avc_rmse_ms']:6.2f} ms | Mean: {metrics['mean_rmse_ms']:6.2f} ms")
    print(f"[R²]   AVO: {metrics['avo_r2']:6.3f}    | AVC: {metrics['avc_r2']:6.3f}    | Mean: {metrics['mean_r2']:6.3f}")
    print(f"[MedAE]AVO: {metrics['avo_medae_ms']:6.2f} ms | AVC: {metrics['avc_medae_ms']:6.2f} ms | Mean: {metrics['mean_medae_ms']:6.2f} ms")
    print(f"[MAX]  AVO: {metrics['avo_max_err_ms']:6.2f} ms | AVC: {metrics['avc_max_err_ms']:6.2f} ms | Mean: {metrics['mean_max_err_ms']:6.2f} ms")
    print(f"[BIAS] AVO: {metrics['avo_bias_ms']:6.2f} ms | AVC: {metrics['avc_bias_ms']:6.2f} ms | Mean: {metrics['mean_bias_ms']:6.2f} ms")
    print(f"[<10ms]AVO: {metrics['avo_acc_10ms_%']:6.1f} %  | AVC: {metrics['avc_acc_10ms_%']:6.1f} %  | Mean: {metrics['mean_acc_10ms_%']:6.1f} %")
    print(f"[<20ms]AVO: {metrics['avo_acc_20ms_%']:6.1f} %  | AVC: {metrics['avc_acc_20ms_%']:6.1f} %  | Mean: {metrics['mean_acc_20ms_%']:6.1f} %")
    print("="*45)


if __name__ == "__main__":
    main()
