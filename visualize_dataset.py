import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    # Load the training dataset
    data_dir = Path("outputs/datasets")
    train_data = np.load(data_dir / "train.npz", allow_pickle=True)
    
    print("Dataset keys:", train_data.files)
    
    assert "y" in train_data.files, "Dataset missing 'y' array"
    assert "target_names" in train_data.files, "Dataset missing 'target_names'"
    
    target_names = list(train_data["target_names"])
    print("Target names:", target_names)
    
    avo_idx_col = target_names.index("avo_ms")
    avc_idx_col = target_names.index("avc_ms")
    
    # Extract the data arrays
    x = train_data["x"]  # Shape: (num_beats, 2 channels, 160 samples)
    y = train_data["y"]  # Shape: (num_beats, num_targets)
    
    # We will plot the first 3 heartbeats
    num_plots = 3
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 10))
    fig.suptitle("Sample Heartbeats from the Dataset (dZ/dt & ECG)", fontsize=16, fontweight='bold')
    
    for i in range(num_plots):
        ax = axes[i]
        
        # Channel 0 is dZ/dt. Channel 1 is ECG.
        dzdt_signal = x[i, 0, :]
        ecg_signal = x[i, 1, :]
        
        # Normalize signals for visualization clarity so they fit on the same plot nicely
        dzdt_norm = (dzdt_signal - np.min(dzdt_signal)) / (np.max(dzdt_signal) - np.min(dzdt_signal))
        ecg_norm = (ecg_signal - np.min(ecg_signal)) / (np.max(ecg_signal) - np.min(ecg_signal))
        
        # In this dataset, the targets are directly in 'y' as the true millisecond values (not normalized)
        # or if they are normalized, we need the raw ones. Wait, the user said:
        # avo_ms = y[i, avo_idx_col]
        # So we will use `y` directly.
        avo_ms = y[i, avo_idx_col]
        avc_ms = y[i, avc_idx_col]
        
        # Convert milliseconds back to sample indices for plotting
        ms_per_sample = 750.0 / 160.0
        r_peak_idx = 250.0 / ms_per_sample
        avo_idx = r_peak_idx + (avo_ms / ms_per_sample)
        avc_idx = r_peak_idx + (avc_ms / ms_per_sample)
        
        # Plot the signal waves
        ax.plot(dzdt_norm, color='black', linewidth=2, label="dZ/dt (Blood Volume Change)")
        ax.plot(ecg_norm, color='gray', linewidth=1, linestyle=':', label="ECG (Electrical)")
        
        # Plot the vertical markers
        ax.axvline(x=r_peak_idx, color='blue', linestyle='--', linewidth=2, label="ECG R-Peak")
        ax.axvline(x=avo_idx, color='green', linestyle='-', linewidth=2, label=f"AVO (+{avo_ms:.1f} ms)")
        ax.axvline(x=avc_idx, color='red', linestyle='-', linewidth=2, label=f"AVC (+{avc_ms:.1f} ms)")
        
        ax.set_ylabel("Normalized Amplitude")
        ax.set_title(f"Sample {i}")
        ax.grid(alpha=0.3)
        
        # Only add the legend to the first plot to avoid clutter
        if i == 0:
            ax.legend(loc="upper right")
            
    axes[-1].set_xlabel("Time (Normalized Samples 0 to 160)")
    plt.tight_layout()
    
    # Save the plot
    output_path = Path("outputs/plots/dataset_visualization.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Dataset visualization saved successfully to: {output_path}")

if __name__ == "__main__":
    main()
