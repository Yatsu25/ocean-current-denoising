"""Petit CNN résiduel de débruitage (inspiré de DnCNN) : le réseau prédit le bruit,
qu'on soustrait à l'entrée."""
from __future__ import annotations

import torch
from torch import nn


class DnCNN(nn.Module):
    def __init__(self, depth: int = 8, width: int = 32):
        super().__init__()
        layers = [nn.Conv2d(1, width, 3, padding=1, padding_mode="reflect"), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(width, width, 3, padding=1, padding_mode="reflect"),
                       nn.BatchNorm2d(width), nn.ReLU(inplace=True)]
        layers.append(nn.Conv2d(width, 1, 3, padding=1, padding_mode="reflect"))
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.body(x)
