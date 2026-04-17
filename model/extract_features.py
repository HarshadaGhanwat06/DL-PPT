from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.cnn_improved import ImprovedCNN


def load_split(data_dir: Path, split_name: str) -> Dict[str, np.ndarray]:
    split = np.load(data_dir / f"{split_name}.npz", allow_pickle=True)
    return {
        "x": split["x"].astype(np.float32),
        "y": split["y_reference"].astype(np.float32) if "y_reference" in split.files else split["y"].astype(np.float32),
        "split": np.asarray([split_name] * len(split["x"])),
    }


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int) -> DataLoader:
    x_tensor = torch.from_numpy(x)
    y_tensor = torch.from_numpy(y)
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


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
    features = []
    targets = []

    model.eval()
    with torch.no_grad():
        for inputs, batch_targets in loader:
            inputs = inputs.to(device)
            embeddings = model.features(inputs).view(inputs.size(0), -1).cpu().numpy()
            features.append(embeddings)
            targets.append(batch_targets.numpy())

    return np.concatenate(features, axis=0), np.concatenate(targets, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract CNN features for the hybrid CNN + XGBoost pipeline.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "outputs" / "datasets" / "dataset_clipped")
    parser.add_argument("--weights", type=Path, default=ROOT / "outputs" / "runs" / "cnn_improved_v2_best_model.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "features")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = load_feature_extractor(args.weights, device)

    split_features = {}
    split_targets = {}
    split_labels = []

    for split_name in ("train", "val", "test"):
        split = load_split(args.data_dir, split_name)
        loader = make_loader(split["x"], split["y"], batch_size=args.batch_size)
        features, targets = extract_split_features(extractor, loader, device)
        split_features[split_name] = features
        split_targets[split_name] = targets
        split_labels.append(np.asarray([split_name] * features.shape[0]))

        np.save(args.output_dir / f"{split_name}_X_features.npy", features)
        np.save(args.output_dir / f"{split_name}_y_targets.npy", targets)

    # Save combined arrays as well so downstream inspection is convenient, while
    # still preserving explicit split files for leakage-safe training.
    X_features = np.concatenate([split_features["train"], split_features["val"], split_features["test"]], axis=0)
    y_targets = np.concatenate([split_targets["train"], split_targets["val"], split_targets["test"]], axis=0)
    split_array = np.concatenate(split_labels, axis=0)

    np.save(args.output_dir / "X_features.npy", X_features)
    np.save(args.output_dir / "y_targets.npy", y_targets)
    np.save(args.output_dir / "split_labels.npy", split_array)

    summary = {
        "weights_path": str(args.weights),
        "feature_dim": int(X_features.shape[1]),
        "num_samples": int(X_features.shape[0]),
        "splits": {name: int(split_features[name].shape[0]) for name in split_features},
    }
    with (args.output_dir / "feature_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
