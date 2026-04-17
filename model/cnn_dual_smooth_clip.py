from __future__ import annotations

import torch
from torch import nn


class _SignalBranch(nn.Module):
    """
    Shared feature extractor used independently for dz/dt and ECG channels.

    The branch is intentionally identical for both signals so the comparison
    stays focused on the target-processing strategies rather than on a changed
    feature extractor.
    """

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


class DualBranchSmoothClipCNN(nn.Module):
    """
    Dual-branch model for the clipped-AVC target experiment.

    Both heads perform regression. The architecture stays close to the earlier
    dual-branch model so the comparison is centered on clipped targets rather
    than on a large architecture change.
    """

    def __init__(self) -> None:
        super().__init__()
        self.dzdt_branch = _SignalBranch()
        self.ecg_branch = _SignalBranch()

        # The shared fusion block lets both branches contribute jointly before
        # the target-specific heads specialize for their own learning problem.
        self.fusion = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
        )

        self.pep_head = nn.Linear(64, 1)
        self.avc_head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dzdt = x[:, 0:1, :]
        ecg = x[:, 1:2, :]
        feature_dzdt = self.dzdt_branch(dzdt)
        feature_ecg = self.ecg_branch(ecg)
        fused_features = torch.cat([feature_dzdt, feature_ecg], dim=1)
        shared_features = self.fusion(fused_features)
        pep_output = self.pep_head(shared_features).squeeze(1)
        avc_output = self.avc_head(shared_features).squeeze(1)
        return pep_output, avc_output
