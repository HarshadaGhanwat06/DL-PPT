from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


def require_torch() -> None:
    if torch is None or nn is None:
        raise ImportError(
            "PyTorch is not installed. Install it in the project virtual environment before training models."
        )


if nn is not None:

    class CNNRegressor(nn.Module):
        def __init__(self, input_length: int, output_dim: int = 2) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=7, padding=3),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(16, 32, kernel_size=5, padding=2),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(16),
            )
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 16, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, output_dim),
            )

        def forward(self, x):
            return self.head(self.features(x))


    class CNNLSTMRegressor(nn.Module):
        def __init__(self, output_dim: int = 2) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=7, padding=3),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(64, 64, kernel_size=5, padding=2),
                nn.ReLU(),
            )
            self.temporal = nn.LSTM(
                input_size=64,
                hidden_size=64,
                num_layers=2,
                batch_first=True,
                dropout=0.2,
                bidirectional=True,
            )
            self.head = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, output_dim),
            )

        def forward(self, x):
            encoded = self.encoder(x)
            encoded = encoded.transpose(1, 2)
            sequence, _ = self.temporal(encoded)
            pooled = sequence.mean(dim=1)
            return self.head(pooled)
