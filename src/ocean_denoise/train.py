"""Entraînement du CNN sur données synthétiques.

Usage : python -m ocean_denoise.train --samples 1200 --epochs 15
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .models import DnCNN
from .synthetic import make_dataset

SCALE = 0.5  # normalisation : l'amplitude typique du signal est ~0.5 m


def to_tensor(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(a / SCALE).float().unsqueeze(1)


def gradients(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Différences finies : les courants géostrophiques sont proportionnels à ces gradients."""
    return x[..., 1:, :] - x[..., :-1, :], x[..., :, 1:] - x[..., :, :-1]


def physics_loss(pred: torch.Tensor, target: torch.Tensor, grad_weight: float) -> torch.Tensor:
    """MSE sur la hauteur de mer + MSE sur ses gradients (donc sur les courants)."""
    loss = torch.nn.functional.mse_loss(pred, target)
    if grad_weight > 0:
        (py, px), (ty, tx) = gradients(pred), gradients(target)
        loss = loss + grad_weight * (torch.nn.functional.mse_loss(py, ty) + torch.nn.functional.mse_loss(px, tx))
    return loss


def train(samples: int = 1200, epochs: int = 15, batch: int = 32, lr: float = 1e-3,
          seed: int = 0, out: str = "results/dncnn.pt", grad_weight: float = 10.0) -> list[float]:
    torch.manual_seed(seed)
    data = make_dataset(samples, seed=seed)
    ds = TensorDataset(to_tensor(data["noisy"]), to_tensor(data["clean"]))
    loader = DataLoader(ds, batch_size=batch, shuffle=True)

    model = DnCNN()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    losses = []
    for ep in range(epochs):
        model.train()
        tot, t0 = 0.0, time.time()
        for x, y in loader:
            opt.zero_grad()
            loss = physics_loss(model(x), y, grad_weight)
            loss.backward()
            opt.step()
            tot += loss.item() * len(x)
        sched.step()
        losses.append(tot / len(ds))
        print(f"époque {ep + 1:2d}/{epochs}  loss={losses[-1]:.5f}  ({time.time() - t0:.1f}s)")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    return losses


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=1200)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--out", default="results/dncnn.pt")
    p.add_argument("--grad-weight", type=float, default=10.0,
                   help="poids du terme de perte sur les gradients (0 = MSE seule)")
    a = p.parse_args()
    train(a.samples, a.epochs, out=a.out, grad_weight=a.grad_weight)
