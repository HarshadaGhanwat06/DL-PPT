from __future__ import annotations

import torch
from torch import nn


class _SignalBranch(nn.Module):
    """Mirror the dual-branch CNN encoder so pretrained weights can be reused."""

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


class CNNFeatureExtractor(nn.Module):
    """
    Dual-branch CNN encoder that returns fused features instead of regression outputs.

    The structure intentionally matches the dual-branch CNN backbone so weights
    from a trained dual-branch checkpoint can be loaded directly and then used
    as a frozen feature extractor for XGBoost.
    """

    def __init__(self, projection_dim: int = 128) -> None:
        super().__init__()
        self.dzdt_branch = _SignalBranch()
        self.ecg_branch = _SignalBranch()
        self.projection = nn.Sequential(
            nn.Linear(128, projection_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dzdt_feat = self.dzdt_branch(x[:, 0:1, :])
        ecg_feat = self.ecg_branch(x[:, 1:2, :])
        combined = torch.cat([dzdt_feat, ecg_feat], dim=1)
        return self.projection(combined)
