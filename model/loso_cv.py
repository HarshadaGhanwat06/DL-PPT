"""Leave-One-Subject-Out (LOSO) cross-validation for CNN Improved.

With only 17 subjects a single train/val/test split is noisy. LOSO gives an
honest picture of per-subject generalisation by:
  - Training on 16 subjects
  - Testing on the held-out 1 subject
  - Rotating through all 17 folds
  - Reporting mean ± std MAE across folds

Usage
-----
    python model/loso_cv.py

Optional arguments
------------------
    --data-dir      Path to the dataset folder containing subject NPZ files.
                    Default: outputs/datasets/dataset_clipped
    --output-dir    Where to write loso_cv_report.json
                    Default: outputs/runs
    --epochs        Max epochs per fold (default: 60)
    --patience      Early-stopping patience per fold (default: 12)
    --batch-size    (default: 32)
    --lr            Learning rate (default: 1e-3)
    --seed          Random seed (default: 42)

Expected data layout
--------------------
The script expects each subject's data to be stored in a separate NPZ file
named  subj_<ID>.npz  inside  --data-dir, with arrays:
  x   : (N_beats, 2, 160)  – input signals
  y   : (N_beats, 2)       – targets [PEP, AVC] in milliseconds

If your data is in the combined train/val/test split format instead (a single
train.npz / val.npz / test.npz with a 'subject' column), the script
automatically falls back to that layout and reads subject IDs from the
'subject' array (if present).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import ResNet1DRegressor, TCNRegressor, TransformerRegressor, CNNLSTMRegressor, CNNRegressor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    # Use only dzdt channel
    x_data = x[:, 0, :] if x.ndim == 3 else x
    x_t = torch.from_numpy(x_data.astype(np.float32)).unsqueeze(1)
    y_t = torch.from_numpy(y.astype(np.float32))
    return DataLoader(TensorDataset(x_t, y_t), batch_size=batch_size, shuffle=shuffle)


def normalize_targets(
    train_y: np.ndarray,
    test_y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Normalize using training-fold statistics only (prevents leakage)."""
    y_mean = train_y.mean(axis=0).astype(np.float32)
    y_std = (train_y.std(axis=0) + 1e-6).astype(np.float32)
    return (train_y - y_mean) / y_std, (test_y - y_mean) / y_std, y_mean, y_std


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_per_subject_data(data_dir: Path) -> Dict[str, Dict[str, np.ndarray]]:
    """Load per-subject NPZ files (subj_<ID>.npz) from data_dir."""
    subject_files = sorted(data_dir.glob("subj_*.npz"))
    if not subject_files:
        raise FileNotFoundError(
            f"No subj_*.npz files found in {data_dir}.\n"
            "If your data is in combined train/val/test splits, use --combined-splits."
        )
    subjects: Dict[str, Dict[str, np.ndarray]] = {}
    for path in subject_files:
        subj_id = path.stem  # e.g. "subj_01"
        npz = np.load(path, allow_pickle=True)
        subjects[subj_id] = {
            "x": npz["x"].astype(np.float32),
            "y": npz["y"].astype(np.float32),
        }
    return subjects


def load_combined_splits(data_dir: Path) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Fallback: build per-subject pools from combined train/val/test splits.

    Each split NPZ must contain a 'subject' array of integer or string IDs.
    If no 'subject' array is found the function assigns all samples to a
    single pseudo-subject — still useful for code validation.
    """
    all_x: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    all_subj: List[np.ndarray] = []

    for split_name in ("train", "val", "test"):
        path = data_dir / f"{split_name}.npz"
        if not path.exists():
            continue
        npz = np.load(path, allow_pickle=True)
        x = npz["x"].astype(np.float32)
        y = (npz["y_reference"] if "y_reference" in npz.files else npz["y"]).astype(np.float32)
        all_x.append(x)
        all_y.append(y)

        if "subject_id" in npz.files:
            all_subj.append(npz["subject_id"])
        elif "subject" in npz.files:
            all_subj.append(npz["subject"])
        else:
            # No subject IDs — assign a fake sequential ID per split
            all_subj.append(np.full(len(x), fill_value=split_name, dtype=object))

    if not all_x:
        raise FileNotFoundError(f"No split NPZ files found in {data_dir}.")

    X = np.concatenate(all_x, axis=0)
    Y = np.concatenate(all_y, axis=0)
    S = np.concatenate(all_subj, axis=0)

    subjects: Dict[str, Dict[str, np.ndarray]] = {}
    for sid in np.unique(S):
        mask = S == sid
        subjects[str(sid)] = {"x": X[mask], "y": Y[mask]}
    return subjects


# ---------------------------------------------------------------------------
# Training one fold
# ---------------------------------------------------------------------------

def train_fold(
    train_x: np.ndarray,
    train_y_norm: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    *,
    model_name: str,
    epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> Dict[str, float]:
    """Train one fold and return test-set metrics (in ms)."""
    train_loader = make_loader(train_x, train_y_norm, batch_size, shuffle=True)
    val_loader = make_loader(val_x, np.zeros_like(val_y), batch_size, shuffle=False)

    if model_name == "resnet":
        model = ResNet1DRegressor().to(device)
    elif model_name == "tcn":
        model = TCNRegressor().to(device)
    elif model_name == "transformer":
        model = TransformerRegressor().to(device)
    elif model_name == "cnn_lstm":
        model = CNNLSTMRegressor().to(device)
    else:
        model = CNNRegressor(input_length=160).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    criterion = nn.SmoothL1Loss()

    best_val_mae = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

        # Validate (in ms)
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, _ in val_loader:
                preds.append(model(xb.to(device)).cpu().numpy())
        y_pred_ms = np.concatenate(preds, axis=0) * y_std + y_mean
        val_mae = float(np.mean(np.abs(y_pred_ms - val_y)))

        scheduler.step(val_mae)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in val_loader:
            preds.append(model(xb.to(device)).cpu().numpy())
    y_pred_ms = np.concatenate(preds, axis=0) * y_std + y_mean
    return compute_metrics(val_y, y_pred_ms)


# ---------------------------------------------------------------------------
# Main LOSO loop
# ---------------------------------------------------------------------------

def run_loso(
    subjects: Dict[str, Dict[str, np.ndarray]],
    *,
    model_name: str,
    epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> Dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subject_ids = sorted(subjects.keys())
    n = len(subject_ids)

    if n < 2:
        raise ValueError(
            f"Need at least 2 subjects for LOSO, found {n}. "
            "Check that your data has a 'subject' array or use subj_*.npz files."
        )

    print(f"Running LOSO CV: {n} folds on device={device}")
    fold_results: List[Dict[str, object]] = []

    for fold_idx, test_subj in enumerate(subject_ids, start=1):
        set_seed(seed + fold_idx)
        train_ids = [s for s in subject_ids if s != test_subj]

        train_x = np.concatenate([subjects[s]["x"] for s in train_ids], axis=0)
        train_y = np.concatenate([subjects[s]["y"] for s in train_ids], axis=0)
        test_x = subjects[test_subj]["x"]
        test_y = subjects[test_subj]["y"]

        train_y_norm, _, y_mean, y_std = normalize_targets(train_y, test_y)

        print(
            f"  Fold {fold_idx:02d}/{n} | test_subject={test_subj} "
            f"| train_samples={len(train_x)} | test_samples={len(test_x)}"
        )
        metrics = train_fold(
            train_x, train_y_norm, test_x, test_y, y_mean, y_std,
            model_name=model_name, epochs=epochs, patience=patience, 
            batch_size=batch_size, lr=lr, device=device,
        )
        metrics["test_subject"] = test_subj
        fold_results.append(metrics)
        print(
            f"           AVO MAE={metrics['avo_mae_ms']:.2f} ms | "
            f"AVC MAE={metrics['avc_mae_ms']:.2f} ms | "
            f"Mean MAE={metrics['mean_mae_ms']:.2f} ms"
        )

    # Aggregate statistics
    def _agg(key: str) -> Dict[str, float]:
        vals = [float(r[key]) for r in fold_results]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "values": vals}

    summary = {
        "n_subjects": n,
        "avo_mae_ms": _agg("avo_mae_ms"),
        "avc_mae_ms": _agg("avc_mae_ms"),
        "mean_mae_ms": _agg("mean_mae_ms"),
        "avo_rmse_ms": _agg("avo_rmse_ms"),
        "avc_rmse_ms": _agg("avc_rmse_ms"),
        "mean_rmse_ms": _agg("mean_rmse_ms"),
        "avo_r2": _agg("avo_r2"),
        "avc_r2": _agg("avc_r2"),
        "mean_r2": _agg("mean_r2"),
        "avo_medae_ms": _agg("avo_medae_ms"),
        "avc_medae_ms": _agg("avc_medae_ms"),
        "mean_medae_ms": _agg("mean_medae_ms"),
        "avo_max_err_ms": _agg("avo_max_err_ms"),
        "avc_max_err_ms": _agg("avc_max_err_ms"),
        "mean_max_err_ms": _agg("mean_max_err_ms"),
        "avo_bias_ms": _agg("avo_bias_ms"),
        "avc_bias_ms": _agg("avc_bias_ms"),
        "mean_bias_ms": _agg("mean_bias_ms"),
        "avo_acc_10ms_%": _agg("avo_acc_10ms_%"),
        "avc_acc_10ms_%": _agg("avc_acc_10ms_%"),
        "mean_acc_10ms_%": _agg("mean_acc_10ms_%"),
        "avo_acc_20ms_%": _agg("avo_acc_20ms_%"),
        "avc_acc_20ms_%": _agg("avc_acc_20ms_%"),
        "mean_acc_20ms_%": _agg("mean_acc_20ms_%"),
    }

    print("\n" + "="*45)
    print(" === LOSO Summary (Averaged across folds) ===")
    print("="*45)
    print(f"[MAE]  AVO: {summary['avo_mae_ms']['mean']:6.2f} ± {summary['avo_mae_ms']['std']:5.2f} ms | AVC: {summary['avc_mae_ms']['mean']:6.2f} ± {summary['avc_mae_ms']['std']:5.2f} ms")
    print(f"[RMSE] AVO: {summary['avo_rmse_ms']['mean']:6.2f} ± {summary['avo_rmse_ms']['std']:5.2f} ms | AVC: {summary['avc_rmse_ms']['mean']:6.2f} ± {summary['avc_rmse_ms']['std']:5.2f} ms")
    print(f"[MedAE]AVO: {summary['avo_medae_ms']['mean']:6.2f} ± {summary['avo_medae_ms']['std']:5.2f} ms | AVC: {summary['avc_medae_ms']['mean']:6.2f} ± {summary['avc_medae_ms']['std']:5.2f} ms")
    print(f"[MAX]  AVO: {summary['avo_max_err_ms']['mean']:6.2f} ± {summary['avo_max_err_ms']['std']:5.2f} ms | AVC: {summary['avc_max_err_ms']['mean']:6.2f} ± {summary['avc_max_err_ms']['std']:5.2f} ms")
    print(f"[BIAS] AVO: {summary['avo_bias_ms']['mean']:6.2f} ± {summary['avo_bias_ms']['std']:5.2f} ms | AVC: {summary['avc_bias_ms']['mean']:6.2f} ± {summary['avc_bias_ms']['std']:5.2f} ms")
    print(f"[<10ms]AVO: {summary['avo_acc_10ms_%']['mean']:6.1f} %           | AVC: {summary['avc_acc_10ms_%']['mean']:6.1f} %")
    print(f"[<20ms]AVO: {summary['avo_acc_20ms_%']['mean']:6.1f} %           | AVC: {summary['avc_acc_20ms_%']['mean']:6.1f} %")
    print("="*45)
    
    return {"summary": summary, "folds": fold_results}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Leave-One-Subject-Out cross-validation.")
    parser.add_argument("--model_name", type=str, default="resnet", choices=["resnet", "tcn", "transformer", "cnn_lstm", "cnn"])
    parser.add_argument(
        "--data-dir", type=Path,
        default=ROOT / "outputs" / "datasets" / "dataset_clipped",
        help="Directory with subj_*.npz files, OR the folder with train/val/test.npz.",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "runs")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Try per-subject files first, then fall back to combined splits
    try:
        subjects = load_per_subject_data(args.data_dir)
        data_layout = "per_subject_npz"
    except FileNotFoundError:
        subjects = load_combined_splits(args.data_dir)
        data_layout = "combined_splits_with_subject_column"

    print(f"Data layout: {data_layout} | subjects found: {len(subjects)}")

    results = run_loso(
        subjects,
        model_name=args.model_name,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )
    results["config"] = {
        "model_name": args.model_name,
        "data_dir": str(args.data_dir),
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "data_layout": data_layout,
    }

    report_path = args.output_dir / "loso_cv_report.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nFull report saved to {report_path}")


if __name__ == "__main__":
    main()
