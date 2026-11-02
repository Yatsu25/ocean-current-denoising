"""Méthode de référence : lissage gaussien (ce qu'on fait classiquement sur une MDT)."""
import numpy as np
from scipy.ndimage import gaussian_filter


def gaussian_denoise(ssh: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    return gaussian_filter(ssh, sigma=sigma, mode="reflect").astype(np.float32)


def best_sigma(noisy: np.ndarray, clean: np.ndarray, sigmas=(0.5, 1.0, 1.5, 2.0, 3.0, 4.0)) -> float:
    """Choisit le sigma qui minimise l'erreur sur un jeu de validation (comparaison équitable)."""
    errs = [np.mean([(gaussian_denoise(n, s) - c) ** 2 for n, c in zip(noisy, clean)]) for s in sigmas]
    return float(sigmas[int(np.argmin(errs))])
