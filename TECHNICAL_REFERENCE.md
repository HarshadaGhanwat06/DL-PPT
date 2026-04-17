# Technical Reference Guide
## Code Implementations & Deep Dives

---

## Table of Contents
1. [Model Architectures](#model-architectures)
2. [Training Loops](#training-loops)
3. [Feature Extraction](#feature-extraction)
4. [Hybrid Inference](#hybrid-inference)
5. [Critical Code Patterns](#critical-code-patterns)
6. [Debugging Guide](#debugging-guide)

---

## Model Architectures

### CNN Improved V2 (Production Model)

**File**: `model/cnn_improved.py`

```python
import torch
from torch import nn

class ImprovedCNN(nn.Module):
    """Deeper 1D CNN regressor for predicting PEP and AVC from dz/dt + ECG."""

    def __init__(
        self,
        input_channels: int = 2,
        output_dim: int = 2,
        dropout: float = 0.5,
        hidden_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        
        # Feature extraction blocks
        self.features = nn.Sequential(
            # Block 1: Input → 32 channels
            nn.Conv1d(input_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            
            # Block 2: 32 → 32 channels (residual-like depth)
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),  # Reduce length by 2
            
            # Block 3: 32 → 64 channels
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            
            # Block 4: 64 → 64 channels
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),  # Reduce length by 2
            
            # Block 5: 64 → 128 channels
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            
            # Block 6: 128 → 128 channels (final conv)
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),  # Global average pooling
        )
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Flatten(),  # From (batch, 128, 1) → (batch, 128)
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=hidden_dropout),
            nn.Linear(64, output_dim),  # Output: [PEP, AVC]
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, 2, 160)
               - Channel 0: dz/dt signal
               - Channel 1: ECG signal (placeholder)
        
        Returns:
            Tensor of shape (batch, 2) with [PEP_pred, AVC_pred]
        """
        return self.regressor(self.features(x))


# Usage
model = ImprovedCNN(input_channels=2, output_dim=2)
x = torch.randn(32, 2, 160)  # Batch of 32 signals
output = model(x)  # Shape: (32, 2)
```

**Layer-by-Layer Analysis**:
```
Input: (batch=32, channels=2, length=160)
  ↓
Conv1d(2→32) + BN + ReLU: (32, 32, 160)
Conv1d(32→32) + BN + ReLU: (32, 32, 160)
MaxPool1d(2): (32, 32, 80)
  ↓
Conv1d(32→64) + BN + ReLU: (32, 64, 80)
Conv1d(64→64) + BN + ReLU: (32, 64, 80)
MaxPool1d(2): (32, 64, 40)
  ↓
Conv1d(64→128) + BN + ReLU: (32, 128, 40)
Conv1d(128→128) + BN + ReLU: (32, 128, 40)
AdaptiveAvgPool1d(1): (32, 128, 1)
  ↓
Flatten: (32, 128)
FC(128→128) + ReLU + Dropout(0.5): (32, 128)
FC(128→64) + ReLU + Dropout(0.3): (32, 64)
FC(64→2): (32, 2) → [PEP, AVC]
```

**Parameter Count**:
```
Conv layers:
  Conv1d(2→32): 2*5*32 + 32 = 352
  Conv1d(32→32): 32*5*32 + 32 = 5,152
  Conv1d(32→64): 32*5*64 + 64 = 10,304
  Conv1d(64→64): 64*5*64 + 64 = 20,544
  Conv1d(64→128): 64*5*128 + 128 = 40,960
  Conv1d(128→128): 128*5*128 + 128 = 81,920

FC layers:
  FC(128→128): 128*128 + 128 = 16,512
  FC(128→64): 128*64 + 64 = 8,256
  FC(64→2): 64*2 + 2 = 130

Total: ~184K parameters
```

---

### Dual-Branch CNN (Baseline for Comparison)

**File**: `model/cnn_dual_smooth_clip.py`

```python
class _SignalBranch(nn.Module):
    """Feature extractor for dZ/dt or ECG signal."""
    
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x):
        # x shape: (batch, 1, 160)
        return self.features(x).flatten(start_dim=1)  # → (batch, 64)


class DualBranchSmoothClipCNN(nn.Module):
    """Dual-branch CNN with separate heads for PEP and AVC."""
    
    def __init__(self) -> None:
        super().__init__()
        self.dzdt_branch = _SignalBranch()      # 64-dim features
        self.ecg_branch = _SignalBranch()       # 64-dim features
        
        # Shared fusion
        self.fusion = nn.Sequential(
            nn.Linear(128, 128),  # 64 + 64 = 128 input
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
        )
        
        # Separate heads
        self.pep_head = nn.Linear(64, 1)
        self.avc_head = nn.Linear(64, 1)

    def forward(self, x):
        # x shape: (batch, 2, 160)
        dzdt = x[:, 0:1, :]      # (batch, 1, 160)
        ecg = x[:, 1:2, :]       # (batch, 1, 160)
        
        feature_dzdt = self.dzdt_branch(dzdt)  # (batch, 64)
        feature_ecg = self.ecg_branch(ecg)     # (batch, 64)
        
        fused = torch.cat([feature_dzdt, feature_ecg], dim=1)  # (batch, 128)
        shared_features = self.fusion(fused)   # (batch, 64)
        
        pep_output = self.pep_head(shared_features).squeeze(1)  # (batch,)
        avc_output = self.avc_head(shared_features).squeeze(1)  # (batch,)
        
        return pep_output, avc_output
```

---

## Training Loops

### CNN Training with Early Stopping

**File**: `model/train_cnn_improved.py`

```python
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch."""
    model.train()
    running_loss = 0.0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        predictions = model(inputs)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item()) * inputs.size(0)

    return running_loss / len(loader.dataset)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    y_mean: np.ndarray,
    y_std: np.ndarray,
) -> Dict[str, object]:
    """Evaluate model on one split."""
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
    
    # Denormalize back to milliseconds
    y_pred = y_pred_norm * y_std + y_mean
    y_true = y_true_norm * y_std + y_mean
    
    metrics = compute_metrics(y_true, y_pred)
    return {
        "metrics": metrics,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def train_with_early_stopping(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    max_epochs: int = 60,
    patience: int = 12,
) -> Dict[str, list]:
    """
    Train model with early stopping based on validation MAE.
    """
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001
    )
    criterion = nn.SmoothL1Loss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    history = {
        "train_loss": [],
        "val_mae_ms": [],
    }
    
    best_val_mae = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(1, max_epochs + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        history["train_loss"].append(train_loss)
        
        # Validate
        val_result = evaluate(
            model, val_loader, device, y_mean, y_std
        )
        val_mae = val_result["metrics"]["mean_mae_ms"]
        history["val_mae_ms"].append(val_mae)
        
        print(f"Epoch {epoch}/{max_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val MAE: {val_mae:.2f} ms")
        
        # Early stopping check
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_model.pt")
            print(f"  ✓ Best model saved")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping after {epoch} epochs")
                model.load_state_dict(torch.load("best_model.pt"))
                break
        
        scheduler.step(val_mae)
    
    return history, model
```

---

## Feature Extraction

### Correct Feature Extraction Pipeline

**File**: `model/extract_features.py`

```python
def load_feature_extractor(weights_path: Path, device: torch.device) -> ImprovedCNN:
    """
    Load the ImprovedCNN model for feature extraction.

    IMPORTANT: Features must be extracted using the SAME CNN used later in hybrid model
    Do NOT mix models (causes feature mismatch)
    """
    model = ImprovedCNN().to(device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def extract_split_features(
    model: ImprovedCNN,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract CNN features for all samples in a split.
    
    Output shape: (num_samples, 128)
    - Each sample is the output of CNN.features() layer
    - Flattened from (batch, 128, 1) → (batch, 128)
    """
    features = []
    targets = []

    model.eval()
    with torch.no_grad():
        for inputs, batch_targets in loader:
            inputs = inputs.to(device)
            
            # Extract intermediate features (CRITICAL: NOT full output)
            embeddings = model.features(inputs)  # (batch, 128, 1)
            embeddings_flat = embeddings.view(inputs.size(0), -1)  # (batch, 128)
            
            features.append(embeddings_flat.cpu().numpy())
            targets.append(batch_targets.numpy())

    return np.concatenate(features, axis=0), np.concatenate(targets, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract CNN features for the hybrid CNN + XGBoost pipeline."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=ROOT / "outputs" / "runs" / "cnn_improved_v2_best_model.pt"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "features"
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    extractor = load_feature_extractor(args.weights, device)

    # Process each split
    split_features = {}
    split_targets = {}
    split_labels = []

    for split_name in ("train", "val", "test"):
        print(f"Extracting {split_name} features...")
        
        split = load_split(data_dir, split_name)
        loader = make_loader(split["x"], split["y"], batch_size=64)
        
        features, targets = extract_split_features(extractor, loader, device)
        split_features[split_name] = features
        split_targets[split_name] = targets
        split_labels.append(np.asarray([split_name] * features.shape[0]))

        # Save individual splits
        np.save(args.output_dir / f"{split_name}_X_features.npy", features)
        np.save(args.output_dir / f"{split_name}_y_targets.npy", targets)
        
        print(f"  {split_name}: {features.shape}")

    # Combine across splits
    X_features = np.concatenate(
        [split_features["train"], split_features["val"], split_features["test"]],
        axis=0
    )
    y_targets = np.concatenate(
        [split_targets["train"], split_targets["val"], split_targets["test"]],
        axis=0
    )
    split_array = np.concatenate(split_labels, axis=0)

    np.save(args.output_dir / "X_features.npy", X_features)
    np.save(args.output_dir / "y_targets.npy", y_targets)
    np.save(args.output_dir / "split_labels.npy", split_array)

    # Save metadata
    summary = {
        "weights_path": str(args.weights),
        "feature_dim": int(X_features.shape[1]),
        "num_samples": int(X_features.shape[0]),
        "splits": {name: int(split_features[name].shape[0]) for name in split_features},
    }
    with (args.output_dir / "feature_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
```

---

## Hybrid Inference

### Final Hybrid Model Inference

**File**: `model/final_hybrid_model.py`

```python
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
        """Load CNN model for both PEP and feature extraction."""
        model = ImprovedCNN().to(self.device)
        state_dict = torch.load(weights_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def _load_pep_normalization(self, report_path: Path) -> tuple[float, float]:
        """Load PEP normalization parameters from training report."""
        with report_path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        return float(report["target_mean_ms"][0]), float(report["target_std_ms"][0])

    def _load_xgb_model(self, model_path: Path):
        """Load XGBoost model for AVC prediction."""
        if XGBRegressor is None:
            raise ImportError("xgboost is not installed")
        model = XGBRegressor()
        model.load_model(model_path)
        return model

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract CNN features for XGBoost."""
        features = self.cnn_model.features(x)
        return features.view(features.size(0), -1)  # Flatten to 2D

    def predict_batch(
        self,
        x: np.ndarray,
        *,
        precomputed_features: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Predict [PEP, AVC] in milliseconds for one batch of inputs.

        PEP comes from the CNN regression head and is denormalized using the
        training statistics. AVC comes from the XGBoost regressor using either
        precomputed features or on-the-fly CNN features.
        """
        x_tensor = torch.from_numpy(x).to(self.device)
        
        with torch.no_grad():
            # PEP prediction from CNN
            cnn_output = self.cnn_model(x_tensor)
            pep_pred_norm = cnn_output[:, 0]  # First output is PEP
            pep_pred_ms = pep_pred_norm.cpu().numpy() * self.pep_std + self.pep_mean

            # Feature extraction for AVC
            if precomputed_features is None:
                features_np = self.extract_features(x_tensor).cpu().numpy()
            else:
                features_np = precomputed_features

        # AVC prediction from XGBoost
        avc_pred_ms = self.avc_model.predict(features_np)
        
        return np.stack([pep_pred_ms, avc_pred_ms], axis=1).astype(np.float32)


# Usage
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cnn-weights",
        type=Path,
        default=ROOT / "outputs" / "runs" / "cnn_improved_v2_best_model.pt"
    )
    parser.add_argument(
        "--cnn-report",
        type=Path,
        default=ROOT / "outputs" / "runs" / "cnn_improved_v2_report.json"
    )
    parser.add_argument(
        "--avc-xgb-model",
        type=Path,
        default=ROOT / "outputs" / "runs" / "xgb_avc_model.json"
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = FinalHybridPredictor(
        cnn_weights_path=args.cnn_weights,
        cnn_report_path=args.cnn_report,
        avc_xgb_model_path=args.avc_xgb_model,
        device=device,
    )

    # Load data
    split = load_split(data_dir, args.split)
    feature_path = feature_dir / f"{args.split}_X_features.npy"
    precomputed_features = np.load(feature_path) if feature_path.exists() else None

    # Predict
    y_pred = predictor.predict_batch(
        split["x"],
        precomputed_features=precomputed_features
    )
    y_true = split["y_true"]
    
    # Evaluate
    metrics = compute_metrics(y_true, y_pred)
    print(json.dumps({"metrics": metrics}, indent=2))
```

---

## Critical Code Patterns

### Pattern 1: Device Management

```python
# CORRECT: Explicit device placement
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)
x = x.to(device)
y = y.to(device)

# WRONG: Implicit placement (can cause CUDA/CPU mismatch)
model(x)  # ❌ If model on GPU and x on CPU: error

# CORRECT: Consistent conversion
x_tensor = torch.from_numpy(x).to(device)
```

---

### Pattern 2: Feature Flattening

```python
# STAGE 1: CNN outputs 3D tensor
features_3d = model.features(x)  # Shape: (batch, 128, 1)

# STAGE 2: Flatten for XGBoost (expects 2D)
features_2d = features_3d.view(features_3d.size(0), -1)  # (batch, 128)

# STAGE 3: Convert to numpy for XGBoost
features_np = features_2d.cpu().numpy()

# STAGE 4: Ensure 2D array for XGBoost predict
assert features_np.ndim == 2, f"Expected 2D, got {features_np.ndim}D"
y_pred = xgb_model.predict(features_np)
```

---

### Pattern 3: Normalization & Denormalization

```python
# NORMALIZATION (training)
y_normalized = (y_raw - y_mean) / y_std
model.train(x, y_normalized)

# DENORMALIZATION (inference)
y_pred_normalized = model.predict(x)
y_pred_ms = y_pred_normalized * y_std + y_mean

# SAVE STATISTICS
report = {
    "target_mean_ms": [pep_mean, avc_mean],
    "target_std_ms": [pep_std, avc_std],
}
```

---

### Pattern 4: Data Loading

```python
def load_split(data_dir: Path, split_name: str) -> Dict[str, np.ndarray]:
    """Load one split with error checking."""
    split_path = data_dir / f"{split_name}.npz"
    
    if not split_path.exists():
        raise FileNotFoundError(f"Missing {split_path}")
    
    split = np.load(split_path, allow_pickle=True)
    
    return {
        "x": split["x"].astype(np.float32),
        "y": split["y_reference"].astype(np.float32) 
             if "y_reference" in split.files 
             else split["y"].astype(np.float32),
        "split": split["split"] if "split" in split.files else None,
    }


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int) -> DataLoader:
    """Create data loader from numpy arrays."""
    x_tensor = torch.from_numpy(x)
    y_tensor = torch.from_numpy(y)
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
```

---

### Pattern 5: Report Generation

```python
def save_report(report: Dict, report_path: Path) -> None:
    """Save structured report with validation."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    required_keys = {
        "config", "target_mean_ms", "target_std_ms",
        "history", "test_metrics", "artifacts"
    }
    
    if not required_keys.issubset(report.keys()):
        raise ValueError(f"Missing keys: {required_keys - set(report.keys())}")
    
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    
    print(f"Report saved: {report_path}")
```

---

## Debugging Guide

### Issue 1: Feature Mismatch in Hybrid Model

**Symptom**: 
```
XGBoost training: MAE 39.42 ms
Hybrid inference: MAE 150+ ms
```

**Root Cause**:
```python
# Training used features from CNN v1
features_train = cnn_v1.features(x)

# Inference uses features from CNN v2 (WRONG)
features_test = cnn_v2.features(x)
```

**Solution**:
```python
# Use SAME model everywhere
# Step 1: Re-extract features with correct model
python model/extract_features.py --weights cnn_v2_model.pt

# Step 2: Retrain XGBoost on new features
python model/train_xgboost.py --feature-dir outputs/features

# Step 3: Test hybrid model
python model/final_hybrid_model.py --split test
```

---

### Issue 2: Dimension Mismatch

**Symptom**:
```
ValueError: shapes are not aligned: idx=1 size=128 vs size=256
```

**Cause**: Features not properly flattened
```python
# WRONG
features = model.features(x)  # Shape: (32, 128, 1)
xgb.predict(features)  # Expects (32, 128)

# CORRECT
features = model.features(x).view(x.size(0), -1)  # Shape: (32, 128)
xgb.predict(features)
```

---

### Issue 3: Device Mismatch

**Symptom**:
```
RuntimeError: Expected all tensors to be on the same device,
but found at least two devices, cuda:0 and cpu!
```

**Solution**:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure all on same device
model = model.to(device)
x = torch.from_numpy(x).to(device)
```

---

### Issue 4: Denormalization Error

**Symptom**:
```
Output range: [0, 1] ms (wrong!)
Expected range: [20, 100] ms (correct)
```

**Cause**: Skipped denormalization
```python
# WRONG
y_pred_norm = model.predict(x)
return y_pred_norm  # Still normalized

# CORRECT
y_pred_norm = model.predict(x)
y_pred_ms = y_pred_norm * y_std + y_mean
return y_pred_ms
```

---

### Issue 5: Evaluation Script Not Finding Files

**Symptom**:
```
FileNotFoundError: outputs/runs/cnn_improved_v2_best_model.pt
```

**Solution**:
```bash
# Check current directory
pwd

# Run from correct location
cd d:\dl-ppt\dl-ppt

# Verify files exist
ls outputs/runs/cnn_improved_v2_*

# Run with explicit paths
python model/final_hybrid_model.py \
  --cnn-weights outputs/runs/cnn_improved_v2_best_model.pt \
  --cnn-report outputs/runs/cnn_improved_v2_report.json
```

---

## Performance Optimization Tips

### Tip 1: Batch Size Tuning

```python
# Memory vs Speed trade-off
batch_sizes = [16, 32, 64, 128]
# 32: Good balance for most cases
# 64: Faster if enough GPU memory
# 16: If running out of memory
```

---

### Tip 2: Early Stopping Parameters

```python
patience = 12     # Check 12 epochs after best val loss
factor = 0.5      # Reduce LR by 50% every 5 epochs
min_lr = 1e-6     # Don't reduce below this

# More patience = longer training, risk of overfitting
# Less patience = may stop too early
```

---

### Tip 3: Feature Dimension Reduction

```python
# If 128 features too many for XGBoost
from sklearn.decomposition import PCA

pca = PCA(n_components=64)
features_reduced = pca.fit_transform(features_train)
# Trade-off: Faster XGBoost, less information

# Our choice: Keep 128 (good balance)
```

---

## Validation Checklist

Before deploying hybrid model:

- [ ] CNN model loads without errors
- [ ] Features extracted with correct shape (N, 128)
- [ ] XGBoost model loads without errors
- [ ] Test set prediction runs without errors
- [ ] Output in correct range (PEP: 20-100, AVC: 250-300)
- [ ] Report generated with all metrics
- [ ] Comparison with standalone CNN shows consistency
- [ ] All artifacts saved to correct paths

