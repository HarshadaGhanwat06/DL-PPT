# cnn_regression.py
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class CNNRegressionConfig:
    train_path: Path = Path('outputs/datasets/train.npz')
    val_path: Path = Path('outputs/datasets/val.npz')
    test_path: Path = Path('outputs/datasets/test.npz')
    runs_dir: Path = Path('outputs/runs')
    plots_dir: Path = Path('outputs/plots')
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.3
    patience: int = 10
    seed: int = 42


def ensure_runtime_dirs(base_dir: Path) -> None:
    runtime_dir = base_dir / '.runtime'
    runtime_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('TMP', str(runtime_dir))
    os.environ.setdefault('TEMP', str(runtime_dir))
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', str(runtime_dir / 'torchinductor'))


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


class HeartbeatDataset(Dataset):
    """Dataset wrapper that normalizes 1D segments using train statistics."""

    def __init__(self, x: np.ndarray, y: np.ndarray, x_mean: np.ndarray, x_std: np.ndarray) -> None:
        self.x = ((x.astype(np.float32) - x_mean) / x_std).astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Extract channel 0 (dZ/dt) and keep the channel dimension -> shape: (1, 160)
        features = torch.from_numpy(self.x[index, 0:1, :])
        targets = torch.from_numpy(self.y[index])
        return features, targets


class CNNRegressor(nn.Module):
    """1D CNN for two-target heartbeat regression: [PEP, AVC]."""

    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.features(x))


class EarlyStopping:
    """Stops training when validation loss stops improving."""

    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best_loss = math.inf
        self.counter = 0
        self.should_stop = False

    def step(self, loss: float) -> bool:
        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


def create_dataloaders(config: CNNRegressionConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_data = load_npz(config.train_path)
    val_data = load_npz(config.val_path)
    test_data = load_npz(config.test_path)

    x_mean = train_data['x'].mean(axis=0, keepdims=True).astype(np.float32)
    x_std = train_data['x'].std(axis=0, keepdims=True).astype(np.float32) + 1e-6

    train_dataset = HeartbeatDataset(train_data['x'], train_data['y'], x_mean, x_std)
    val_dataset = HeartbeatDataset(val_data['x'], val_data['y'], x_mean, x_std)
    test_dataset = HeartbeatDataset(test_data['x'], test_data['y'], x_mean, x_std)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, optimizer: torch.optim.Optimizer | None = None) -> float:
    is_training = optimizer is not None
    model.train(is_training)

    running_loss = 0.0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        if is_training:
            optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        if is_training:
            loss.backward()
            optimizer.step()

        running_loss += float(loss.item()) * inputs.size(0)

    return running_loss / len(loader.dataset)


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for inputs, batch_targets in loader:
            inputs = inputs.to(device)
            outputs = model(inputs).cpu().numpy()
            predictions.append(outputs)
            targets.append(batch_targets.numpy())
    return np.concatenate(predictions, axis=0), np.concatenate(targets, axis=0)


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


def plot_loss_curves(history: Dict[str, list[float]], output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_validation_mae(history: Dict[str, list[float]], output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(history['val_mae'], label='Mean Val MAE', linewidth=2)
    plt.plot(history['val_avo_mae'], label='PEP Val MAE', linewidth=1.6)
    plt.plot(history['val_avc_mae'], label='AVC Val MAE', linewidth=1.6)
    plt.xlabel('Epoch')
    plt.ylabel('Validation MAE (ms)')
    plt.title('CNN Regression Validation MAE')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    target_names = ['PEP', 'AVC']
    for index, axis in enumerate(axes):
        axis.scatter(y_true[:, index], y_pred[:, index], alpha=0.6, s=18)
        min_value = min(float(y_true[:, index].min()), float(y_pred[:, index].min()))
        max_value = max(float(y_true[:, index].max()), float(y_pred[:, index].max()))
        axis.plot([min_value, max_value], [min_value, max_value], 'r--', linewidth=1.5)
        axis.set_title(f'{target_names[index]}: Predicted vs True')
        axis.set_xlabel('True')
        axis.set_ylabel('Predicted')
        axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_error_histogram(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    errors = y_pred - y_true
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    target_names = ['PEP Error', 'AVC Error']
    for index, axis in enumerate(axes):
        axis.hist(errors[:, index], bins=30, alpha=0.8, edgecolor='black')
        axis.set_title(target_names[index])
        axis.set_xlabel('Prediction Error')
        axis.set_ylabel('Count')
        axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_report(config: CNNRegressionConfig, history: Dict[str, list[float]], metrics: Dict[str, float], output_path: Path) -> None:
    report = {
        'config': {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        'history': history,
        'test_metrics': metrics,
    }
    with output_path.open('w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)


def train_cnn_regressor(config: CNNRegressionConfig) -> Dict[str, object]:
    config.runs_dir.mkdir(parents=True, exist_ok=True)
    config.plots_dir.mkdir(parents=True, exist_ok=True)
    ensure_runtime_dirs(config.runs_dir.parent)
    set_seed(config.seed)

    prefix = 'cnn_regression'
    plot_subdir = config.plots_dir / prefix
    plot_subdir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, test_loader = create_dataloaders(config)

    model = CNNRegressor(dropout=config.dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    early_stopping = EarlyStopping(patience=config.patience)

    history = {'train_loss': [], 'val_loss': [], 'val_mae': [], 'val_avo_mae': [], 'val_avc_mae': []}
    best_model_path = config.runs_dir / 'standalone_cnn_best.pt'
    report_path = config.runs_dir / 'standalone_cnn_report.json'
    loss_curve_path = plot_subdir / f'{prefix}_loss_curve.png'
    val_mae_curve_path = plot_subdir / f'{prefix}_val_mae_curve.png'
    predictions_path = plot_subdir / f'{prefix}_predicted_vs_true.png'
    error_histogram_path = plot_subdir / f'{prefix}_error_histogram.png'

    existing = [
        str(path)
        for path in (best_model_path, report_path, loss_curve_path, val_mae_curve_path, predictions_path, error_histogram_path)
        if path.exists()
    ]
    print(f'[DEBUG] Plot output folder: {plot_subdir}')
    if existing:
        print('[DEBUG] Existing artifacts detected. This run will overwrite them:')
        for path in existing:
            print(f'[DEBUG]   {path}')
    else:
        print('[DEBUG] No existing artifacts found. New files will be created.')

    for epoch in range(1, config.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        val_loss = run_epoch(model, val_loader, criterion, device, optimizer=None)
        val_pred, val_true = predict(model, val_loader, device)
        val_metrics = compute_metrics(val_true, val_pred)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_metrics['mean_mae_ms'])
        history['val_avo_mae'].append(val_metrics['avo_mae_ms'])
        history['val_avc_mae'].append(val_metrics['avc_mae_ms'])

        print(
            f'Epoch {epoch:02d}/{config.epochs} | '
            f'Train Loss: {train_loss:.4f} | '
            f'Val Loss: {val_loss:.4f} | '
            f'Val MAE: {val_metrics["mean_mae_ms"]:.4f} ms'
        )

        improved = early_stopping.step(val_loss)
        if improved:
            torch.save(model.state_dict(), best_model_path)
            print(f'  Saved best model to {best_model_path}')

        if early_stopping.should_stop:
            print(f'Early stopping triggered at epoch {epoch}.')
            break

    model.load_state_dict(torch.load(best_model_path, map_location=device))

    y_pred, y_true = predict(model, test_loader, device)
    metrics = compute_metrics(y_true, y_pred)

    plot_loss_curves(history, loss_curve_path)
    plot_validation_mae(history, val_mae_curve_path)
    plot_predictions(y_true, y_pred, predictions_path)
    plot_error_histogram(y_true, y_pred, error_histogram_path)
    save_report(config, history, metrics, report_path)

    return {
        'metrics': metrics,
        'history': history,
        'best_model_path': str(best_model_path),
        'report_path': str(report_path),
        'plots_dir': str(plot_subdir),
        'runs_dir': str(config.runs_dir),
        'loss_curve_path': str(loss_curve_path),
        'val_mae_curve_path': str(val_mae_curve_path),
        'predictions_path': str(predictions_path),
        'error_histogram_path': str(error_histogram_path),
    }
