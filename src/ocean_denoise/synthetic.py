"""Génération de champs synthétiques de hauteur de mer (SSH / topographie dynamique)
et des courants géostrophiques associés, avec un bruit réaliste.

Le champ "propre" combine :
  - une pente grande échelle (gradient régional),
  - un courant de bord / jet modélisé par un front en tanh qui méandre,
  - des tourbillons gaussiens (cycloniques et anticycloniques).

Le bruit combine un bruit blanc et un bruit corrélé en bandes (type erreur
d'orbite / traces satellites), ce qui est le cas difficile pour un simple filtre.
"""
from __future__ import annotations

import numpy as np

G = 9.81          # gravité (m/s²)
F_CORIOLIS = 1e-4  # paramètre de Coriolis à moyenne latitude (s⁻¹)


def clean_ssh(n: int = 64, rng: np.random.Generator | None = None,
              regime: str = "energetic", dx_km: float = 10.0) -> np.ndarray:
    """Renvoie un champ SSH propre (en mètres) de taille n x n.

    regime="energetic" : jet + nombreux tourbillons (type Gulf Stream).
    regime="calm"      : peu de tourbillons, faible amplitude (type gyre subtropical).
    """
    rng = rng or np.random.default_rng()
    y, x = np.mgrid[0:n, 0:n] * dx_km
    L = n * dx_km
    ssh = 0.05 * (x / L) + rng.normal(0, 0.02) * (y / L)

    if regime == "energetic":
        y0 = L * rng.uniform(0.35, 0.65)
        meander = 0.08 * L * np.sin(2 * np.pi * x / (L * rng.uniform(0.5, 1.0)) + rng.uniform(0, 2 * np.pi))
        width = rng.uniform(40, 80)
        ssh += 0.5 * np.tanh((y - y0 - meander) / width)
        n_eddies, amp, radius = rng.integers(4, 9), (0.15, 0.35), (25, 60)
    elif regime == "calm":
        n_eddies, amp, radius = rng.integers(1, 4), (0.03, 0.10), (30, 80)
    else:
        raise ValueError(f"régime inconnu : {regime}")

    for _ in range(n_eddies):
        cx, cy = rng.uniform(0, L, size=2)
        r = rng.uniform(*radius)
        a = rng.uniform(*amp) * rng.choice([-1, 1])
        ssh += a * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * r ** 2))
    return ssh.astype(np.float32)


def add_noise(ssh: np.ndarray, rng: np.random.Generator | None = None,
              white_std: float = 0.03, stripe_std: float = 0.04) -> np.ndarray:
    """Ajoute un bruit blanc + un bruit en bandes corrélé le long d'une direction."""
    rng = rng or np.random.default_rng()
    n = ssh.shape[0]
    noise = rng.normal(0, white_std, ssh.shape)
    stripes = rng.normal(0, stripe_std, n)
    stripes = np.convolve(stripes, np.ones(3) / 3, mode="same")
    noise += stripes[None, :] if rng.random() < 0.5 else stripes[:, None]
    return (ssh + noise).astype(np.float32)


def geostrophic_currents(ssh: np.ndarray, dx_km: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """Courants géostrophiques (m/s) : u = -g/f dη/dy, v = g/f dη/dx."""
    dx = dx_km * 1e3
    deta_dy, deta_dx = np.gradient(ssh, dx)
    u = -G / F_CORIOLIS * deta_dy
    v = G / F_CORIOLIS * deta_dx
    return u.astype(np.float32), v.astype(np.float32)


def make_dataset(n_samples: int, n: int = 64, seed: int = 0,
                 regimes: tuple[str, ...] = ("energetic", "calm")) -> dict[str, np.ndarray]:
    """Crée un jeu (bruité, propre, régime) reproductible."""
    rng = np.random.default_rng(seed)
    clean, noisy, labels = [], [], []
    for i in range(n_samples):
        regime = regimes[i % len(regimes)]
        c = clean_ssh(n, rng, regime)
        clean.append(c)
        noisy.append(add_noise(c, rng))
        labels.append(regime)
    return {"clean": np.stack(clean), "noisy": np.stack(noisy), "regime": np.array(labels)}
