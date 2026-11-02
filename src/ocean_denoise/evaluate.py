"""Évaluation sur un jeu de test indépendant, par régime océanique.

Compare : champ bruité / filtre gaussien (sigma optimisé) / CNN.
Usage : python -m ocean_denoise.evaluate --model results/dncnn.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .baselines import best_sigma, gaussian_denoise
from .metrics import evaluate
from .models import DnCNN
from .synthetic import geostrophic_currents, make_dataset
from .train import SCALE, to_tensor


def cnn_denoise(model: DnCNN, noisy: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return (model(to_tensor(noisy)).squeeze(1).numpy() * SCALE).astype(np.float32)


def run(model_path: str, n_test: int = 200, out_dir: str = "results") -> dict:
    val = make_dataset(100, seed=123)
    sigma = best_sigma(val["noisy"], val["clean"])
    test = make_dataset(n_test, seed=999)

    model = DnCNN()
    model.load_state_dict(torch.load(model_path))
    preds = {"bruité": test["noisy"],
             f"gaussien (σ={sigma})": np.stack([gaussian_denoise(x, sigma) for x in test["noisy"]]),
             "CNN": cnn_denoise(model, test["noisy"])}

    results = {}
    for regime in ("energetic", "calm"):
        idx = np.where(test["regime"] == regime)[0]
        results[regime] = {}
        for name, p in preds.items():
            scores = [evaluate(p[i], test["clean"][i]) for i in idx]
            results[regime][name] = {k: round(float(np.mean([s[k] for s in scores])), 4) for k in scores[0]}

    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    _figure(test, preds, out / "exemple.png")
    return results


def _figure(test, preds, path):
    i = int(np.where(test["regime"] == "energetic")[0][0])
    fields = [("propre", test["clean"][i])] + [(k, v[i]) for k, v in preds.items()]
    fig, axes = plt.subplots(2, len(fields), figsize=(3.2 * len(fields), 6.2))
    vmin, vmax = test["clean"][i].min(), test["clean"][i].max()
    for j, (name, f) in enumerate(fields):
        axes[0, j].imshow(f, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        axes[0, j].set_title(f"SSH — {name}")
        u, v = geostrophic_currents(f)
        axes[1, j].imshow(np.hypot(u, v), cmap="viridis", vmin=0, vmax=1.5)
        axes[1, j].set_title("|courant| (m/s)")
        for ax in axes[:, j]:
            ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="results/dncnn.pt")
    a = p.parse_args()
    print(json.dumps(run(a.model), indent=2, ensure_ascii=False))
