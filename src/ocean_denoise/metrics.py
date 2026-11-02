"""Métriques : erreur sur la hauteur de mer, erreur sur les courants, préservation des structures."""
from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity

from .synthetic import geostrophic_currents


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def current_rmse(pred_ssh: np.ndarray, clean_ssh: np.ndarray) -> float:
    """RMSE (m/s) sur le vecteur courant géostrophique."""
    up, vp = geostrophic_currents(pred_ssh)
    uc, vc = geostrophic_currents(clean_ssh)
    return float(np.sqrt(np.mean((up - uc) ** 2 + (vp - vc) ** 2)))


def eke_ratio(pred_ssh: np.ndarray, clean_ssh: np.ndarray) -> float:
    """Rapport d'énergie cinétique : 1 = énergie conservée, <1 = structures lissées, >1 = bruit résiduel."""
    up, vp = geostrophic_currents(pred_ssh)
    uc, vc = geostrophic_currents(clean_ssh)
    return float(np.mean(up ** 2 + vp ** 2) / np.mean(uc ** 2 + vc ** 2))


def ssim(pred: np.ndarray, clean: np.ndarray) -> float:
    rng = float(clean.max() - clean.min()) or 1.0
    return float(structural_similarity(clean, pred, data_range=rng))


def evaluate(pred: np.ndarray, clean: np.ndarray) -> dict[str, float]:
    return {"rmse_ssh_cm": 100 * rmse(pred, clean),
            "rmse_courant_cm_s": 100 * current_rmse(pred, clean),
            "ratio_eke": eke_ratio(pred, clean),
            "ssim": ssim(pred, clean)}
