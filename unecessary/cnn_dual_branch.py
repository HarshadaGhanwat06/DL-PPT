from __future__ import annotations

import torch
from torch import nn


class _SignalBranch(nn.Module):
    """Feature extractor for a single physiological signal channel."""

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
        return self.features(x).flatten(start_dim=1)


class DualBranchCNN(nn.Module):
    """Dual-branch CNN that learns dz/dt and ECG features separately before fusion."""

    def __init__(self, output_dim: int = 2) -> None:
        super().__init__()
        self.dzdt_branch = _SignalBranch()
        self.ecg_branch = _SignalBranch()
        self.head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        dzdt = x[:, 0:1, :]
        ecg = x[:, 1:2, :]
        feature_dzdt = self.dzdt_branch(dzdt)
        feature_ecg = self.ecg_branch(ecg)
        features = torch.cat([feature_dzdt, feature_ecg], dim=1)
        return self.head(features)
