from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import CNNLSTMRegressor, CNNRegressor
from train import _load_split

def load_model(model_name: str, model_path: Path, input_length: int):
    if model_name == "cnn":
        model = CNNRegressor(input_length=input_length)
    elif model_name == "cnn_lstm":
        model = CNNLSTMRegressor()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize training results and model performance.")
    parser.add_argument("--model", choices=("cnn", "cnn_lstm"), default="cnn_lstm")
    parser.add_argument("--report", type=Path, help="Path to the JSON report file")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "plots")
    args = parser.parse_args()

    # Determine paths
    report_path = args.report or ROOT / "outputs" / "runs" / f"{args.model}_report.json"
    model_path = ROOT / "outputs" / "runs" / f"{args.model}.pt"
    dataset_path = ROOT / "outputs" / "datasets"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not report_path.exists():
        print(f"Error: Report file not found at {report_path}")
        return

    # 1. Load Report and Plot Loss History
    with open(report_path, "r") as f:
        report = json.load(f)

    history = report["history"]
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_mae = [h["mean_mae_ms"] for h in history]

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="Train Loss (SmoothL1)")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_mae, label="Val MAE (ms)", color="orange")
    plt.title("Validation Mean MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE (ms)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    loss_plot = args.output_dir / f"{args.model}_loss_history.png"
    plt.savefig(loss_plot)
    print(f"Saved loss plot to {loss_plot}")

    # 2. Run Inference for Predicted vs True
    test_data = _load_split(dataset_path, "test")
    x_test = torch.tensor(test_data["x"], dtype=torch.float32).unsqueeze(1)
    y_true = test_data["y"] # (N, 2) [avo_ms, avc_ms]

    model = load_model(args.model, model_path, input_length=x_test.shape[2])
    
    with torch.no_grad():
        # Handle normalization using values from report
        target_mean = np.array(report["target_mean_ms"])
        target_std = np.array(report["target_std_ms"])
        
        outputs = model(x_test).numpy()
        y_pred = outputs * target_std + target_mean

    # 3. Plot Predicted vs True
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    targets = ["AVO (Aortic Valve Opening)", "AVC (Aortic Valve Closure)"]
    
    for i in range(2):
        axes[i].scatter(y_true[:, i], y_pred[:, i], alpha=0.5, s=10)
        # Identity line
        min_val = min(y_true[:, i].min(), y_pred[:, i].min())
        max_val = max(y_true[:, i].max(), y_pred[:, i].max())
        axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', label="Perfect")
        
        axes[i].set_title(f"Predicted vs True: {targets[i]}")
        axes[i].set_xlabel("True Value (ms)")
        axes[i].set_ylabel("Predicted Value (ms)")
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()

    plt.tight_layout()
    pred_plot = args.output_dir / f"{args.model}_predictions.png"
    plt.savefig(pred_plot)
    print(f"Saved predictions plot to {pred_plot}")

    # 4. Plot Error Histogram
    errors = y_pred - y_true
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for i in range(2):
        axes[i].hist(errors[:, i], bins=30, color="skyblue", edgecolor="black", alpha=0.7)
        axes[i].axvline(0, color="red", linestyle="--")
        axes[i].set_title(f"Error Distribution: {targets[i]}")
        axes[i].set_xlabel("Error (Predicted - True) [ms]")
        axes[i].set_ylabel("Frequency")
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    error_plot = args.output_dir / f"{args.model}_error_histogram.png"
    plt.savefig(error_plot)
    print(f"Saved error histogram to {error_plot}")

if __name__ == "__main__":
    main()
