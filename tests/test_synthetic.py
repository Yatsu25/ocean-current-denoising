import numpy as np

from ocean_denoise.synthetic import add_noise, clean_ssh, geostrophic_currents, make_dataset


def test_shapes_and_reproducibility():
    a = make_dataset(4, n=32, seed=1)
    b = make_dataset(4, n=32, seed=1)
    assert a["clean"].shape == (4, 32, 32)
    assert np.allclose(a["noisy"], b["noisy"])


def test_noise_is_added():
    rng = np.random.default_rng(0)
    c = clean_ssh(32, rng)
    n = add_noise(c, rng)
    assert 0.01 < np.std(n - c) < 0.2


def test_geostrophy_on_linear_slope():
    # pente de 1 cm / 10 km vers l'est -> courant vers le nord v = g/f * 1e-6 ≈ 0.098 m/s
    x = np.arange(16, dtype=np.float32)
    ssh = np.tile(0.01 * x, (16, 1))
    u, v = geostrophic_currents(ssh, dx_km=10.0)
    assert np.allclose(u, 0, atol=1e-6)
    assert np.allclose(v, 9.81 / 1e-4 * 1e-6, rtol=1e-3)
