# DL-PPT: Heartbeat Event Prediction from ICG Signals

This repository builds a complete deep learning pipeline for predicting cardiac timing events from heartbeat-level impedance cardiography (ICG) segments.

The project uses the HeartCycle HDF5 dataset, extracts synchronized physiological signals, prepares beat-level training samples, and trains regression models to estimate cardiac event timings.

## Project Goal

Given one heartbeat segment of processed ICG (`dZ/dt`), predict cardiac timing targets such as:

- `AVO` / `PEP`: timing of aortic valve opening relative to ECG R-peak
- `AVC`: timing of aortic valve closure relative to ECG R-peak
- derived systolic intervals such as `LVET`

## Dataset

The raw dataset is stored in `data/` as HDF5 (`.h5`) files.

Signals used in this project:

- ECG
- raw ICG impedance signal
- ECG R-peaks
- AVO labels
- AVC labels
- PEP
- LVET

Important HDF5 signal IDs:

- `_030` -> ECG
- `_031` -> raw impedance / ICG
- `_032` -> R-peaks
- `_033` -> AVO
- `_034` -> PEP / PPEjec
- `_035` -> AVC
- `_036` -> LVET

## Current Dataset Summary

Based on the current inspection and preparation pipeline:

- 212 HDF5 records were found
- 17 subjects are included
- beat-level windows are resampled to length `160`
- subject-wise train/validation/test splitting is used
- generated dataset files are stored in `outputs/datasets/`

Prepared dataset files:

- `outputs/datasets/train.npz`
- `outputs/datasets/val.npz`
- `outputs/datasets/test.npz`
- `outputs/datasets/all_segments.npz`
- `outputs/datasets/summary.json`

## Preprocessing Pipeline

The dataset preparation pipeline does the following:

1. Load all `.h5` files
2. Read ECG, ICG, R-peaks, AVO, AVC, PEP, LVET
3. Convert raw ICG to `dZ/dt`
4. Smooth and normalize the signal
5. Segment heartbeats using ECG R-peaks
6. Align each segment with corresponding targets
7. Filter invalid beats using physiological rules
8. Save train, validation, and test splits

## Implemented Models

### 1. Baseline Model

`CNNRegressor` in `models.py`

- 1D convolutional neural network
- learns waveform features from beat segments
- used as the baseline deep learning model

### 2. Main Model

`CNNLSTMRegressor` in `models.py`

- CNN feature extractor + LSTM temporal model
- designed to capture both waveform shape and sequential heartbeat dynamics
- this is the primary model for the project

### 3. Standalone CNN Experiment

Files in `model/`:

- `model/cnn_regression.py`
- `model/train_cnn_regression.py`

This is a separate CNN-only regression pipeline that also produces:

- loss curve
- predicted vs true plot
- error histogram
- saved CNN weights

Its generated outputs are stored in:

- `outputs/runs/` for `.pt` and `.json` artifacts
- `outputs/plots/` for generated plots

## Repository Structure

```text
DL-PPT/
|-- data/                     # raw HeartCycle HDF5 files
|-- outputs/
|   `-- datasets/            # prepared train/val/test NPZ files
|-- model/                   # standalone CNN regression experiment
|-- scripts/
|   |-- inspect_dataset.py   # inspect dataset structure and label quality
|   |-- prepare_dataset.py   # create beat-level NPZ datasets
|   |-- train_model.py       # train baseline CNN or CNN+LSTM
|   `-- visualize_results.py # helper visualization script
|-- data.py                  # data loading, preprocessing, segmentation
|-- models.py                # CNN and CNN+LSTM model definitions
|-- train.py                 # main training and evaluation pipeline
|-- dataset_test.py          # quick dataset sanity check
`-- README.md
```

## Main Files Explained

### `data.py`

Contains the full dataset processing logic:

- HDF5 loading
- raw ICG -> `dZ/dt`
- heartbeat segmentation
- label extraction
- quality filtering
- dataset split creation

### `models.py`

Defines the main project models:

- `CNNRegressor`
- `CNNLSTMRegressor`

### `train.py`

Contains the training loop for the main project pipeline:

- dataset loading from `outputs/datasets/`
- model selection
- target normalization
- optimization
- validation
- early stopping behavior through best-model tracking
- test evaluation
- report saving

### `scripts/train_model.py`

Command-line entry point for training the main models.

## How to Run

### 1. Inspect dataset

```powershell
.\.venv\Scripts\python.exe scripts\inspect_dataset.py
```

### 2. Prepare dataset

```powershell
.\.venv\Scripts\python.exe scripts\prepare_dataset.py
```

### 3. Train baseline CNN

```powershell
.\.venv\Scripts\python.exe scripts\train_model.py --model cnn --epochs 10
```

### 4. Train main CNN + LSTM model

```powershell
.\.venv\Scripts\python.exe scripts\train_model.py --model cnn_lstm --epochs 15
```

### 5. Run standalone CNN regression experiment

```powershell
.\.venv\Scripts\python.exe model\train_cnn_regression.py
```

## Outputs

### Main training pipeline outputs

Saved under `outputs/runs/` when `scripts/train_model.py` is run:

- `cnn.pt`
- `cnn_report.json`
- `cnn_lstm.pt`
- `cnn_lstm_report.json`

These reports contain:

- configuration used for training
- train / validation / test metrics
- baseline comparison metrics
- target normalization statistics
- per-epoch history

### Standalone CNN outputs

Saved under `outputs/runs/`:

- `standalone_cnn_best.pt`
- `standalone_cnn_report.json`

Saved under `outputs/plots/`:

- `standalone_cnn_loss_curve.png`
- `standalone_cnn_predicted_vs_true.png`
- `standalone_cnn_error_histogram.png`

## Evaluation Metrics

Models are evaluated using:

- MAE
- RMSE

for both regression targets.

## Environment

This project is currently set up with a local virtual environment in `.venv`.

Typical Python libraries used:

- `numpy`
- `h5py`
- `matplotlib`
- `torch`
- `scikit-learn`
- `scipy`

## Notes

- Paths in the scripts have been updated to resolve from the project root.
- Subject-wise splitting is used to reduce leakage between train and test sets.
- Dataset filtering removes physiologically invalid beats before training.
- `dataset_test.py` is only for dataset sanity checking and does not contain model code.

## Future Improvements

Possible next steps:

- add Transformer-based time-series model
- improve baseline signal heuristics
- add richer evaluation plots for main models
- export predictions beat-by-beat for error analysis
- clean remaining debug prints in `data.py`
