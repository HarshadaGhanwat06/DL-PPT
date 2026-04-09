from __future__ import annotations

import os
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict

import numpy as np

from models import CNNLSTMRegressor, CNNRegressor, require_torch

require_torch()
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainingConfig:
    data_dir: Path = Path("outputs/datasets")
    output_dir: Path = Path("outputs/runs")
    model_name: str = "cnn_lstm"
    seed: int = 42
    epochs: int = 25
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 5


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


def _make_loader(
    split_data: Dict[str, np.ndarray],
    batch_size: int,
    shuffle: bool,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> DataLoader:
    x = torch.tensor(split_data["x"], dtype=torch.float32).unsqueeze(1)
    y = torch.tensor((split_data["y"] - target_mean) / target_std, dtype=torch.float32)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)


def _build_model(model_name: str, input_length: int):
    if model_name == "cnn":
        return CNNRegressor(input_length=input_length)
    if model_name == "cnn_lstm":
        return CNNLSTMRegressor()
    raise ValueError(f"Unsupported model_name={model_name!r}")


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    error = y_pred - y_true
    mae = np.mean(np.abs(error), axis=0)
    rmse = np.sqrt(np.mean(np.square(error), axis=0))
    return {
        "avo_mae_ms": float(mae[0]),
        "avc_mae_ms": float(mae[1]),
        "avo_rmse_ms": float(rmse[0]),
        "avc_rmse_ms": float(rmse[1]),
        "mean_mae_ms": float(mae.mean()),
        "mean_rmse_ms": float(rmse.mean()),
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


def _baseline_metrics(split_data: Dict[str, np.ndarray]) -> Dict[str, float]:
    return _compute_metrics(split_data["y"], split_data["baseline"])


def train_model(config: TrainingConfig) -> Dict[str, object]:
    _set_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _ensure_runtime_dirs(config.output_dir.parent)

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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.SmoothL1Loss()

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
            optimizer.zero_grad()
            predictions = model(x_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * x_batch.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_metrics = _evaluate_model(model, val_loader, device, target_mean=target_mean, target_std=target_std)
        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss,
            **val_metrics,
        }
        history.append(epoch_result)

        if val_metrics["mean_mae_ms"] < best_val_mae:
            best_val_mae = val_metrics["mean_mae_ms"]
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_metrics = _evaluate_model(model, train_loader, device, target_mean=target_mean, target_std=target_std)
    val_metrics = _evaluate_model(model, val_loader, device, target_mean=target_mean, target_std=target_std)
    test_metrics = _evaluate_model(model, test_loader, device, target_mean=target_mean, target_std=target_std)
    test_baseline = _baseline_metrics(test_data)

    torch.save(model.state_dict(), config.output_dir / f"{config.model_name}.pt")

    report: Dict[str, object] = {
        "config": _config_to_jsonable_dict(config),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "baseline_test_metrics": test_baseline,
        "target_mean_ms": target_mean.tolist(),
        "target_std_ms": target_std.tolist(),
        "history": history,
    }
    with (config.output_dir / f"{config.model_name}_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report
