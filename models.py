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
            # head input = hidden_size * 2 (forward + backward final hidden state)
            self.head = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, output_dim),
            )

        def forward(self, x):
            encoded = self.encoder(x)
            encoded = encoded.transpose(1, 2)  # (B, T, C)
            # hn shape: (num_layers * num_directions, B, hidden_size)
            _, (hn, _) = self.temporal(encoded)
            # hn[-2] = last forward layer, hn[-1] = last backward layer
            pooled = torch.cat([hn[-2], hn[-1]], dim=1)  # (B, 128)
            return self.head(pooled)


    class ResBlock1D(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.conv1 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
            self.bn1 = nn.BatchNorm1d(channels)
            self.conv2 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
            self.bn2 = nn.BatchNorm1d(channels)
            self.relu = nn.ReLU()
        def forward(self, x):
            res = x
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.bn2(self.conv2(x))
            return self.relu(x + res)

    class ResNet1DRegressor(nn.Module):
        def __init__(self, output_dim: int = 2):
            super().__init__()
            self.in_conv = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=7, padding=3),
                nn.BatchNorm1d(32),
                nn.ReLU()
            )
            self.blocks = nn.Sequential(
                ResBlock1D(32),
                nn.MaxPool1d(2),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                ResBlock1D(64),
                nn.MaxPool1d(2),
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                ResBlock1D(128),
                nn.AdaptiveAvgPool1d(1)
            )
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, output_dim)
            )
        def forward(self, x):
            return self.head(self.blocks(self.in_conv(x)))

    class TCNBlock(nn.Module):
        def __init__(self, in_c: int, out_c: int, dilation: int):
            super().__init__()
            self.conv1 = nn.Conv1d(in_c, out_c, kernel_size=3, padding=dilation, dilation=dilation)
            self.bn1 = nn.BatchNorm1d(out_c)
            self.conv2 = nn.Conv1d(out_c, out_c, kernel_size=3, padding=dilation, dilation=dilation)
            self.bn2 = nn.BatchNorm1d(out_c)
            self.relu = nn.ReLU()
            self.proj = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

        def forward(self, x):
            res = self.proj(x)
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.bn2(self.conv2(x))
            return self.relu(x + res)

    class TCNRegressor(nn.Module):
        def __init__(self, output_dim: int = 2):
            super().__init__()
            self.blocks = nn.Sequential(
                TCNBlock(1, 32, dilation=1),
                TCNBlock(32, 64, dilation=2),
                TCNBlock(64, 64, dilation=4),
                TCNBlock(64, 128, dilation=8),
                TCNBlock(128, 128, dilation=16),
                nn.AdaptiveAvgPool1d(1)
            )
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, output_dim)
            )
        def forward(self, x):
            return self.head(self.blocks(x))

    class TransformerRegressor(nn.Module):
        def __init__(self, output_dim: int = 2):
            super().__init__()
            self.in_proj = nn.Conv1d(1, 64, kernel_size=5, padding=2)
            self.pos_emb = nn.Parameter(torch.zeros(1, 160, 64))
            encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, dropout=0.1, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
            self.head = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, output_dim)
            )

        def forward(self, x):
            x = self.in_proj(x)
            x = x.transpose(1, 2)
            x = x + self.pos_emb
            x = self.transformer(x)
            x = x.mean(dim=1)
            return self.head(x)
