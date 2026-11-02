import numpy as np
import torch

from ocean_denoise.baselines import gaussian_denoise
from ocean_denoise.metrics import evaluate
from ocean_denoise.models import DnCNN


def test_model_keeps_shape():
    m = DnCNN(depth=4, width=8)
    x = torch.randn(2, 1, 32, 32)
    assert m(x).shape == x.shape


def test_metrics_perfect_prediction():
    rng = np.random.default_rng(0)
    c = rng.normal(size=(32, 32)).astype(np.float32)
    s = evaluate(c, c)
    assert s["rmse_ssh_cm"] == 0 and abs(s["ratio_eke"] - 1) < 1e-6 and s["ssim"] > 0.999


def test_gaussian_reduces_white_noise():
    rng = np.random.default_rng(0)
    c = np.zeros((32, 32), np.float32)
    n = c + rng.normal(0, 0.1, c.shape).astype(np.float32)
    assert np.std(gaussian_denoise(n, 1.5)) < np.std(n)


def test_physics_loss_penalises_gradients():
    from ocean_denoise.train import physics_loss
    t = torch.zeros(1, 1, 8, 8)
    p = torch.zeros(1, 1, 8, 8)
    p[..., ::2, :] = 0.1  # même erreur moyenne qu'un biais, mais beaucoup de gradients
    assert physics_loss(p, t, 10.0) > physics_loss(p, t, 0.0)
