# DL-PPT: Heartbeat Event Prediction from ICG Signals

DL-PPT is a research-oriented pipeline for estimating cardiac timing parameters from synchronized physiological waveforms. The project evolved from a single-channel convolutional baseline into multi-branch physiological models, subject-independent validation, and a final hybrid deep learning plus gradient boosting workflow.

The two main targets are:

- `PEP` (Pre-Ejection Period)
- `AVC` (Aortic Valve Closure)

## Project Overview

The repository builds beat-level learning pipelines from raw HeartCycle HDF5 recordings. Each heartbeat is aligned to the ECG R-peak, converted into fixed-length segments, and used for regression of cardiac event timings in milliseconds.

Signals used in the final pipeline:

- `ECG`
- raw impedance cardiography (`ICG`)
- derived `dZ/dt`
- R-peaks
- AVO / PEP labels
- AVC labels
- LVET labels

The final dataset format is centered on dual-channel inputs:

- channel 0: `dZ/dt`
- channel 1: `ECG`

with target pairs defined in milliseconds.

## Data Processing and Methodology

### Signal Extraction

Raw physiological signals are loaded from `.h5` files in `data/`. The loader reads ECG, raw ICG, R-peaks, AVO, AVC, PEP, and LVET annotations, then derives subject IDs for subject-wise splitting.

### dZ/dt Transformation

The raw impedance signal is transformed into `dZ/dt` using a numerical derivative, followed by baseline correction and smoothing. This highlights rapid mechanical changes that are more informative for valve event prediction than raw impedance alone.

### Beat Segmentation

Each sample is extracted around an ECG R-peak using a fixed temporal window:

- `250 ms` before the R-peak
- `500 ms` after the R-peak

Both ECG and `dZ/dt` are resampled to length `160`, producing fixed-size heartbeat representations suitable for neural models.

### Normalization and Filtering

Each beat segment is normalized independently to reduce amplitude variation across subjects. The pipeline also applies physiological validity checks to remove implausible beats before training. In later experiments, clipped target variants were introduced to reduce the effect of noisy AVC labels.

### Target Formulation

The project uses two target formulations:

- direct regression: `[PEP, AVC]`
- physiological reformulation: `[PEP, LVET]`, with `AVC = PEP + LVET`

This reformulation was important because it reduced target redundancy and improved physiological interpretability.

## Finalized Models

### Baseline CNN Regression Model

Purpose: establish a benchmark architecture for direct cardiac event regression.

Main characteristics:

- single-channel `dZ/dt` input
- direct prediction of `PEP` and `AVC`
- `Conv1D + BatchNorm + MaxPool + Fully Connected` layers
- `MSE` loss

This model provides the simplest reference point for comparing later architectural and physiological improvements.

### Improved CNN

Purpose: strengthen feature extraction and training stability beyond the baseline.

Main characteristics:

- dual-channel `ECG + dZ/dt` input
- deeper convolutional blocks
- batch normalization after convolution layers
- adaptive / global average pooling
- dropout regularization
- `SmoothL1Loss`

This model improved training stability and feature learning, and it became the main backbone for later hybrid experiments. In the saved `cnn_improved_v2` results, it delivered the strongest standalone `PEP` performance among the finalized CNN variants.

### Smooth-Clipped Dual CNN

Purpose: improve robustness to noisy physiological labels, especially for `AVC`.

Main characteristics:

- dual-branch architecture with separate signal processing paths
- direct regression of `PEP` and `AVC`
- clipped targets for more robust learning
- weighted `SmoothL1` / Huber-style objective

This model was designed to reduce the impact of extreme `AVC` labels while preserving end-to-end regression.

### Advanced Physiological Dual CNN

Purpose: improve physiological learning by reformulating the target space.

Main characteristics:

- dual-branch architecture with richer shared feature learning
- separate output heads
- predicts `PEP` and `LVET`
- reconstructs `AVC` using `AVC = PEP + LVET`
- `Log-Cosh` loss

This model reduced direct dependence on noisy `AVC` supervision and made the regression problem more physiologically meaningful.

### ResNet Model

Purpose: test deeper convolutional learning with better optimization behavior.

Main characteristics:

- residual skip connections
- deeper stable learning
- improved gradient flow

The residual architecture was introduced to explore whether depth alone could improve event prediction without destabilizing optimization.

### TCN (Temporal Convolutional Network)

Purpose: capture long-range temporal structure more effectively than standard CNNs.

Main characteristics:

- dilated temporal convolutions
- expanded receptive field without recurrent layers
- strong temporal dependency modeling

Among the advanced architectures, the TCN showed the strongest temporal modeling behavior and the best research potential for subject-independent evaluation.

### Transformer Model

Purpose: model global temporal relationships using self-attention.

Main characteristics:

- self-attention encoder architecture
- global temporal dependency modeling
- flexible sequence representation learning

In practice, the Transformer showed overfitting tendencies because the dataset remains relatively small at the subject level, making it less reliable than the best convolutional and physiological models.

## Hybrid CNN + XGBoost Framework

The final hybrid design separates the two tasks by using the most suitable model family for each target:

- CNN for `PEP` prediction and deep feature extraction
- XGBoost for `AVC` regression

Pipeline:

```text
ECG + dZ/dt
    ↓
Improved CNN
    ↓
Deep Feature Extraction
    ↓
XGBoost
    ↓
AVC Prediction
```

Motivation:

- CNNs are well suited for precise temporal pattern learning, which benefits `PEP`
- gradient boosting is more robust to noisy nonlinear targets, which benefits `AVC`
- the hybrid design keeps the representation learning power of deep models while improving regression robustness

The final corrected hybrid pipeline enforces feature consistency:

- the same `cnn_improved_v2` backbone is used for feature extraction and hybrid inference
- old mixed-backbone feature reuse is avoided
- train / validation / test separation is preserved throughout feature extraction and XGBoost training

## Training Strategy

The finalized training pipelines use the following core strategy:

- `Adam` or `AdamW` optimization depending on the model branch
- early stopping based on validation performance
- learning-rate scheduling with `ReduceLROnPlateau`
- batch normalization for stable internal activations
- dropout for regularization
- weight decay to reduce overfitting

Later models also introduced:

- weighted losses to emphasize harder `AVC`-related targets
- robust losses such as `SmoothL1Loss` and `Log-Cosh`
- target engineering through clipping and physiological reformulation

## Validation Strategy

Subject-independent evaluation is critical because the dataset contains repeated beats from a limited number of subjects. The project therefore uses Leave-One-Subject-Out Cross-Validation (`LOSO-CV`) for advanced evaluation.

LOSO-CV procedure:

1. Hold out one subject as the test fold.
2. Train on the remaining `N - 1` subjects.
3. Evaluate on the unseen subject.
4. Repeat for every subject.
5. Aggregate metrics across folds.

Purpose:

- prevents subject leakage
- measures generalization to unseen physiology
- provides a more realistic evaluation than a single random split

## Evaluation Metrics

The project reports both error magnitude and calibration-oriented metrics.

Let `y_i` be the ground-truth target and `ŷ_i` the prediction for `n` samples.

### Mean Absolute Error (MAE)

```text
MAE = (1 / n) * Σ |y_i - ŷ_i|
```

Measures average absolute timing error in milliseconds.

### Root Mean Squared Error (RMSE)

```text
RMSE = sqrt((1 / n) * Σ (y_i - ŷ_i)^2)
```

Penalizes larger errors more strongly than MAE.

### Coefficient of Determination (R²)

```text
R² = 1 - (Σ (y_i - ŷ_i)^2 / Σ (y_i - ȳ)^2)
```

Measures how much target variance is explained by the model.

### Median Absolute Error (MedAE)

```text
MedAE = median(|y_i - ŷ_i|)
```

Provides a robust central error measure that is less sensitive to outliers.

### Bias

```text
Bias = (1 / n) * Σ (ŷ_i - y_i)
```

Measures systematic overestimation or underestimation.

### Accuracy Within ±10 ms

```text
Acc_10 = (count(|y_i - ŷ_i| <= 10) / n) * 100
```

Measures the percentage of predictions falling within `±10 ms` of ground truth.

### Accuracy Within ±20 ms

```text
Acc_20 = (count(|y_i - ŷ_i| <= 20) / n) * 100
```

Measures the percentage of predictions falling within `±20 ms` of ground truth.

## Experimental Findings

The finalized experiments show a consistent pattern:

- the improved CNN substantially stabilized feature extraction and training compared with the original baseline CNN
- the physiological dual-head formulation improved `PEP` learning by predicting `LVET` instead of direct `AVC`
- clipped-target training improved robustness to noisy physiological labels
- the TCN provided the strongest temporal modeling behavior among the advanced deep architectures
- the Transformer underperformed because the available subject count is too small for stable attention-heavy modeling
- the hybrid CNN + XGBoost design demonstrated a useful modular regression approach, especially for handling noisy `AVC`
- physiological target engineering contributed more than simply increasing architecture complexity

Representative saved results in `outputs/runs/` also reflect this tradeoff:

- `cnn_improved_v2` produced the strongest standalone `PEP` result among finalized CNN variants
- `xgb_report.json` shows XGBoost remained competitive for `AVC`, supporting the hybrid design motivation

## Repository Structure

```text
DL-PPT/
|-- data/                           # Raw HeartCycle HDF5 files
|-- outputs/
|   |-- datasets/                   # Prepared train/val/test datasets
|   |-- features/                   # CNN feature arrays for XGBoost
|   |-- plots/                      # Saved visualizations
|   `-- runs/                       # Models and JSON reports
|-- model/
|   |-- cnn_improved.py             # Improved dual-channel CNN
|   |-- cnn_dual_smooth_clip.py     # Smooth-clipped dual-branch CNN
|   |-- cnn_dual_advanced.py        # Physiological PEP + LVET model
|   |-- cnn_feature_extractor.py    # CNN backbone for hybrid features
|   |-- extract_features.py         # Hybrid feature extraction
|   |-- train_xgboost.py            # XGBoost training on CNN features
|   |-- final_hybrid_model.py       # Final hybrid inference pipeline
|   `-- loso_cv.py                  # Leave-One-Subject-Out evaluation
|-- scripts/
|   |-- inspect_dataset.py
|   |-- prepare_dataset.py
|   |-- train_model.py
|   `-- visualize_results.py
|-- data.py                         # Data extraction and preprocessing
|-- models.py                       # Baseline, ResNet, TCN, Transformer
|-- train.py                        # Main training/evaluation pipeline
`-- README.md
```

## How to Run

Prepare the dataset:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_dataset.py
```

Train the main baseline-style models:

```powershell
.\.venv\Scripts\python.exe scripts\train_model.py --model cnn
.\.venv\Scripts\python.exe scripts\train_model.py --model cnn_lstm
```

Train the improved CNN:

```powershell
.\.venv\Scripts\python.exe model\train_cnn_improved.py
```

Train the clipped dual-branch model:

```powershell
.\.venv\Scripts\python.exe model\train_cnn_dual_smooth_clip.py --data-dir outputs/datasets/dataset_clipped
```

Train the advanced physiological model:

```powershell
.\.venv\Scripts\python.exe model\train_cnn_dual_advanced.py
```

Extract features and train XGBoost:

```powershell
.\.venv\Scripts\python.exe model\extract_features.py --split train
.\.venv\Scripts\python.exe model\extract_features.py --split val
.\.venv\Scripts\python.exe model\extract_features.py --split test
.\.venv\Scripts\python.exe model\train_xgboost.py --feature-dir outputs/features
```

Run the final hybrid model:

```powershell
.\.venv\Scripts\python.exe model\final_hybrid_model.py --split test
```

## Final Conclusion

The final outcome of this project is not simply that deeper models perform better. Instead, the experiments show that target engineering and physiological reformulation were more impactful than blindly increasing architecture complexity.

The strongest research directions emerging from the project are:

- physiological dual-branch modeling with explicit `PEP` / `LVET` structure
- TCN-based temporal modeling
- subject-independent evaluation via `LOSO-CV`

Overall, LOSO-CV provided the most realistic evaluation protocol, and the combination of physiological insight plus robust learning design proved more valuable than architecture scaling alone.
