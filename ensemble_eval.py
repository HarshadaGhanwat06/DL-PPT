"""Ensemble the top models for maximum performance.

As per Item 6, ensembling different architectures reliably beats any 
single model by 3-8% with zero extra architectural work. We simply run
the test set through ResNet, TCN, and Transformer, and average their predictions.
"""

import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from models import ResNet1DRegressor, TCNRegressor, TransformerRegressor
from train import _compute_metrics

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load the test set
    data_dir = Path("outputs/datasets")
    test_data = np.load(data_dir / "test.npz", allow_pickle=True)
    train_data = np.load(data_dir / "train.npz", allow_pickle=True)
    
    target_mean = train_data["y"].mean(axis=0).astype(np.float32)
    target_std = train_data["y"].std(axis=0).astype(np.float32) + 1e-6
    
    # Extract only dzdt (channel 0)
    x_test = test_data["x"][:, 0, :]
    x_tensor = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1)
    
    # 2. Initialize models and load weights
    models = {
        "resnet": ResNet1DRegressor().to(device),
        "tcn": TCNRegressor().to(device),
        "transformer": TransformerRegressor().to(device),
    }
    
    runs_dir = Path("outputs/runs")
    
    for name, model in models.items():
        weight_path = runs_dir / f"{name}.pt"
        if not weight_path.exists():
            print(f"ERROR: Could not find {weight_path}. Make sure you trained the {name} model.")
            return
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.eval()

    # 3. Predict and Average
    print(f"Generating predictions on {len(x_test)} test samples...")
    all_preds = []
    
    with torch.no_grad():
        x_batch = x_tensor.to(device)
        
        preds_resnet = models["resnet"](x_batch).cpu().numpy()
        preds_tcn = models["tcn"](x_batch).cpu().numpy()
        preds_transformer = models["transformer"](x_batch).cpu().numpy()
        
    # Stack and average (Ensemble)
    stacked_preds = np.stack([preds_resnet, preds_tcn, preds_transformer])
    ensemble_preds_norm = np.mean(stacked_preds, axis=0)
    
    # De-normalize predictions
    y_true = test_data["y"]
    ensemble_preds = ensemble_preds_norm * target_std + target_mean
    
    # 4. Compute Metrics
    metrics = _compute_metrics(y_true, ensemble_preds)
    
    print("="*45)
    print(" ENSEMBLE TEST METRICS (ResNet+TCN+Transformer) ")
    print("="*45)
    
    # 1. MAE
    print(f"[MAE]  AVO: {metrics['avo_mae_ms']:6.2f} ms | AVC: {metrics['avc_mae_ms']:6.2f} ms | Mean: {metrics['mean_mae_ms']:6.2f} ms")
    # 2. RMSE
    print(f"[RMSE] AVO: {metrics['avo_rmse_ms']:6.2f} ms | AVC: {metrics['avc_rmse_ms']:6.2f} ms | Mean: {metrics['mean_rmse_ms']:6.2f} ms")
    # 3. R2 Score
    print(f"[R²]   AVO: {metrics['avo_r2']:6.3f}    | AVC: {metrics['avc_r2']:6.3f}    | Mean: {metrics['mean_r2']:6.3f}")
    # 4. MedAE
    print(f"[MedAE]AVO: {metrics['avo_medae_ms']:6.2f} ms | AVC: {metrics['avc_medae_ms']:6.2f} ms | Mean: {metrics['mean_medae_ms']:6.2f} ms")
    # 5. Max Error
    print(f"[MAX]  AVO: {metrics['avo_max_err_ms']:6.2f} ms | AVC: {metrics['avc_max_err_ms']:6.2f} ms | Mean: {metrics['mean_max_err_ms']:6.2f} ms")
    # 6. Bias
    print(f"[BIAS] AVO: {metrics['avo_bias_ms']:6.2f} ms | AVC: {metrics['avc_bias_ms']:6.2f} ms | Mean: {metrics['mean_bias_ms']:6.2f} ms")
    # 7. Acc 10ms
    print(f"[<10ms]AVO: {metrics['avo_acc_10ms_%']:6.1f} %  | AVC: {metrics['avc_acc_10ms_%']:6.1f} %  | Mean: {metrics['mean_acc_10ms_%']:6.1f} %")
    # 8. Acc 20ms
    print(f"[<20ms]AVO: {metrics['avo_acc_20ms_%']:6.1f} %  | AVC: {metrics['avc_acc_20ms_%']:6.1f} %  | Mean: {metrics['mean_acc_20ms_%']:6.1f} %")
    print("="*45)
    
    # Save the ensemble report
    report = {
        "models_ensembled": list(models.keys()),
        "metrics": metrics
    }
    
    with open(runs_dir / "ensemble_report.json", "w") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    main()
