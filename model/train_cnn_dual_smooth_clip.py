# train_cnn_dual_smooth_clip.py
from __future__ import annotations

import argparse
import json
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

from model.cnn_dual_smooth_clip import DualBranchSmoothClipCNN

W_PEP = 0.4
W_AVC = 0.6


@dataclass
class SmoothClipConfig:
    data_dir: Path = Path("outputs/datasets/dataset_clipped")
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


def load_data(data_dir: Path) -> Dict[str, object]:
    """
    Load the clipped target dataset.

    The dataset now stores:
    - `y`: the clipped training target
    - `y_reference`: the clean regression target used for evaluation
    This separation lets the trainer keep using the clipped labels while still
    evaluating against explicit reference targets.
    """
    data_dir = Path(data_dir)
    with (data_dir / "summary.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    target_variant = summary["target_variant"]
    if target_variant != "clipped":
        raise ValueError(
            f"Expected dataset_clipped contents, but found target_variant={target_variant!r}."
        )

    splits: Dict[str, Dict[str, np.ndarray]] = {}
    for split_name in ("train", "val", "test"):
        split = np.load(data_dir / f"{split_name}.npz", allow_pickle=True)
        x = split["x"].astype(np.float32)
        if x.ndim != 3 or x.shape[1] != 2:
            raise ValueError(
                f"Expected {split_name}.npz x to have shape (N, 2, 160), got {x.shape!r}."
            )
        splits[split_name] = {
            "x": x,
            "y": split["y"].astype(np.float32),
            "y_reference": split["y_reference"].astype(np.float32),
        }

    return {
        "target_variant": target_variant,
        "splits": splits,
    }


def normalize_pep_targets(
    train_y: np.ndarray,
    val_y: np.ndarray,
    test_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Normalize only the PEP target.

    AVC remains in clipped millisecond space so the loss stays directly tied to
    the physically meaningful timing target used by this experiment.
    """
    pep_mean = float(train_y[:, 0].mean())
    pep_std = float(train_y[:, 0].std() + 1e-6)

    train_norm = train_y.copy()
    val_norm = val_y.copy()
    test_norm = test_y.copy()
    train_norm[:, 0] = (train_norm[:, 0] - pep_mean) / pep_std
    val_norm[:, 0] = (val_norm[:, 0] - pep_mean) / pep_std
    test_norm[:, 0] = (test_norm[:, 0] - pep_mean) / pep_std
    return train_norm, val_norm, test_norm, pep_mean, pep_std


def make_loader(x: np.ndarray, y: np.ndarray, y_reference: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    """
    Package model targets and clean reference targets together.

    The trainer uses:
    - `y` for the optimization objective
    - `y_reference` for denormalized MAE/RMSE evaluation
    """
    x_tensor = torch.from_numpy(x)
    y_tensor = torch.from_numpy(y)
    y_reference_tensor = torch.from_numpy(y_reference)
    dataset = TensorDataset(x_tensor, y_tensor, y_reference_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def build_model() -> DualBranchSmoothClipCNN:
    return DualBranchSmoothClipCNN()


def compute_loss(
    *,
    pep_pred: torch.Tensor,
    avc_pred: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, float, float]:
    """
    Compute the task-specific loss.

    The clipped experiment uses weighted SmoothL1Loss on PEP and AVC so AVC
    still receives the larger optimization focus.
    """
    loss_pep = nn.functional.smooth_l1_loss(pep_pred, targets[:, 0])
    loss_avc = nn.functional.smooth_l1_loss(avc_pred, targets[:, 1])
    loss = W_PEP * loss_pep + W_AVC * loss_avc
    return loss, float(loss_pep.item()), float(loss_avc.item())


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0

    for inputs, targets, _ in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        pep_pred, avc_pred = model(inputs)
        loss, _, _ = compute_loss(
            pep_pred=pep_pred,
            avc_pred=avc_pred,
            targets=targets,
        )
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
    pep_mean: float,
    pep_std: float,
) -> Dict[str, object]:
    """
    Evaluate using clean reference targets in [PEP, AVC] space.
    """
    model.eval()
    pep_predictions = []
    avc_predictions = []
    reference_targets = []

    with torch.no_grad():
        for inputs, _, batch_reference in loader:
            inputs = inputs.to(device)
            pep_pred, avc_pred = model(inputs)

            pep_pred_ms = pep_pred.cpu().numpy() * pep_std + pep_mean
            pep_predictions.append(pep_pred_ms)
            avc_predictions.append(avc_pred.cpu().numpy())
            reference_targets.append(batch_reference.numpy())

    pep_pred_all = np.concatenate(pep_predictions, axis=0)
    avc_pred_all = np.concatenate(avc_predictions, axis=0)
    y_true = np.concatenate(reference_targets, axis=0)
    y_pred = np.stack([pep_pred_all, avc_pred_all], axis=1).astype(np.float32)
    metrics = compute_metrics(y_true, y_pred)
    return {
        "metrics": metrics,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def plot_results(
    history: Dict[str, list[float]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_paths: Dict[str, Path],
    prefix: str,
) -> Dict[str, str]:
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title(f"{prefix} Training Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_paths["loss_curve_path"], dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history["val_mae_ms"], label="Mean Val MAE", linewidth=2)
    plt.plot(history["val_avo_mae_ms"], label="PEP Val MAE", linewidth=1.6)
    plt.plot(history["val_avc_mae_ms"], label="AVC Val MAE", linewidth=1.6)
    plt.xlabel("Epoch")
    plt.ylabel("Validation MAE (ms)")
    plt.title(f"{prefix} Validation MAE")
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


def ensure_output_paths_available(config: SmoothClipConfig, prefix: str) -> Dict[str, Path]:
    plot_subdir = config.plots_dir / prefix
    plot_subdir.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "best_model_path": config.runs_dir / f"{prefix}_best_model.pt",
        "report_path": config.runs_dir / f"{prefix}_report.json",
        "loss_curve_path": plot_subdir / f"{prefix}_loss_curve.png",
        "val_mae_curve_path": plot_subdir / f"{prefix}_val_mae_curve.png",
        "predicted_vs_true_path": plot_subdir / f"{prefix}_predicted_vs_true.png",
        "error_histogram_path": plot_subdir / f"{prefix}_error_histogram.png",
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


def train_and_evaluate(config: SmoothClipConfig) -> Dict[str, object]:
    config.runs_dir.mkdir(parents=True, exist_ok=True)
    config.plots_dir.mkdir(parents=True, exist_ok=True)
    ensure_runtime_dirs(config.runs_dir.parent)
    set_seed(config.seed)

    data = load_data(config.data_dir)
    target_variant = data["target_variant"]
    prefix = "cnn_dual_smooth_clipped"
    output_paths = ensure_output_paths_available(config, prefix)

    train_split = data["splits"]["train"]
    val_split = data["splits"]["val"]
    test_split = data["splits"]["test"]

    train_y_norm, val_y_norm, test_y_norm, pep_mean, pep_std = normalize_pep_targets(
        train_split["y"],
        val_split["y"],
        test_split["y"],
    )

    train_loader = make_loader(train_split["x"], train_y_norm, train_split["y_reference"], config.batch_size, True)
    val_loader = make_loader(val_split["x"], val_y_norm, val_split["y_reference"], config.batch_size, False)
    test_loader = make_loader(test_split["x"], test_y_norm, test_split["y_reference"], config.batch_size, False)

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
        val_result = evaluate(
            model,
            val_loader,
            device,
            pep_mean,
            pep_std,
        )
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
        raise RuntimeError("Training finished without producing a best checkpoint.")

    model.load_state_dict(best_state)
    train_result = evaluate(model, train_loader, device, pep_mean, pep_std)
    val_result = evaluate(model, val_loader, device, pep_mean, pep_std)
    test_result = evaluate(model, test_loader, device, pep_mean, pep_std)
    plot_paths = plot_results(history, test_result["y_true"], test_result["y_pred"], output_paths, prefix)

    report = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "target_variant": target_variant,
        "target_names": ["pep_ms", "avc_ms"],
        "history": [
            {
                "epoch": index + 1,
                "train_loss": history["train_loss"][index],
                "val_mae_ms": history["val_mae_ms"][index],
                "val_avo_mae_ms": history["val_avo_mae_ms"][index],
                "val_avc_mae_ms": history["val_avc_mae_ms"][index],
            }
            for index in range(len(history["train_loss"]))
        ],
        "pep_normalization": {
            "mean": pep_mean,
            "std": pep_std,
        },
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
            "avc": W_AVC,
        },
    }

    with output_paths["report_path"].open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the dual-branch target-noise experiments.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "outputs" / "datasets" / "dataset_clipped")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "outputs" / "runs")
    parser.add_argument("--plots-dir", type=Path, default=ROOT / "outputs" / "plots")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = SmoothClipConfig(
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
    print(f"[MAE]  AVO: {test_metrics['avo_mae_ms']:6.2f} ms | AVC: {test_metrics['avc_mae_ms']:6.2f} ms | Mean: {test_metrics['mean_mae_ms']:6.2f} ms")
    print(f"[RMSE] AVO: {test_metrics['avo_rmse_ms']:6.2f} ms | AVC: {test_metrics['avc_rmse_ms']:6.2f} ms | Mean: {test_metrics['mean_rmse_ms']:6.2f} ms")
    print(f"[R²]   AVO: {test_metrics['avo_r2']:6.3f}    | AVC: {test_metrics['avc_r2']:6.3f}    | Mean: {test_metrics['mean_r2']:6.3f}")
    print(f"[MedAE]AVO: {test_metrics['avo_medae_ms']:6.2f} ms | AVC: {test_metrics['avc_medae_ms']:6.2f} ms | Mean: {test_metrics['mean_medae_ms']:6.2f} ms")
    print(f"[MAX]  AVO: {test_metrics['avo_max_err_ms']:6.2f} ms | AVC: {test_metrics['avc_max_err_ms']:6.2f} ms | Mean: {test_metrics['mean_max_err_ms']:6.2f} ms")
    print(f"[BIAS] AVO: {test_metrics['avo_bias_ms']:6.2f} ms | AVC: {test_metrics['avc_bias_ms']:6.2f} ms | Mean: {test_metrics['mean_bias_ms']:6.2f} ms")
    print(f"[<10ms]AVO: {test_metrics['avo_acc_10ms_%']:6.1f} %  | AVC: {test_metrics['avc_acc_10ms_%']:6.1f} %  | Mean: {test_metrics['mean_acc_10ms_%']:6.1f} %")
    print(f"[<20ms]AVO: {test_metrics['avo_acc_20ms_%']:6.1f} %  | AVC: {test_metrics['avc_acc_20ms_%']:6.1f} %  | Mean: {test_metrics['mean_acc_20ms_%']:6.1f} %")
    print("="*45)


if __name__ == "__main__":
    main()
