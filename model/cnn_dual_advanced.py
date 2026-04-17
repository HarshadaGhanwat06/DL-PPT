from __future__ import annotations

import torch
from torch import nn


class _SignalBranch(nn.Module):
    """Extract a compact feature vector from one physiological channel."""

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).flatten(start_dim=1)


class Attention(nn.Module):
    """Feature-wise attention over the fused branch representation."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, dim),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.attn(x)
        return x * weights


class _RegressionHead(nn.Module):
    """Independent regression head for one cardiac timing target."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DualBranchAdvancedCNN(nn.Module):
    """
    Dual-branch CNN with attention and separate heads for PEP and LVET.

    The model learns dz/dt and ECG features independently, fuses them, applies
    feature-wise attention, and then predicts PEP and LVET with specialized
    regression heads. AVC can later be reconstructed as PEP + LVET.
    """

    def __init__(self) -> None:
        super().__init__()
        self.dzdt_branch = _SignalBranch()
        self.ecg_branch = _SignalBranch()
        self.attention = Attention(dim=128)
        self.pep_head = _RegressionHead()
        self.lvet_head = _RegressionHead()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dzdt = x[:, 0:1, :]
        ecg = x[:, 1:2, :]
        feature_dzdt = self.dzdt_branch(dzdt)
        feature_ecg = self.ecg_branch(ecg)
        fused_features = torch.cat([feature_dzdt, feature_ecg], dim=1)
        attended_features = self.attention(fused_features)
        pep_out = self.pep_head(attended_features)
        lvet_out = self.lvet_head(attended_features)
        return torch.cat([pep_out, lvet_out], dim=1)
