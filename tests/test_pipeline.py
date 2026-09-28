"""
Базовые тесты пайплайна
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from tsprep.config import ALL_MODEL_NAMES, DataConfig, ModelConfig
from tsprep.data_preprocessing.datasets import chronological_split_indices, prepare_and_window, sliding_windows
from tsprep.data_preprocessing.preprocessing import TrainFittedNormalizer, difference, inverse_difference
from tsprep.data_preprocessing.synthetic import generate_series
from tsprep.models.autoencoder import build_model


def test_sliding_windows_shape():
    x = np.arange(100).reshape(100, 1).astype(float)
    w = sliding_windows(x, window_len=10, stride=2)
    assert w.shape == (46, 10, 1)
    assert np.array_equal(w[0, :, 0], x[:10, 0])
    assert np.array_equal(w[1, :, 0], x[2:12, 0])


def test_chronological_split_no_overlap():
    train_end, val_end = chronological_split_indices(1000, 0.7, 0.15, 0.15)
    assert train_end == 700
    assert 700 < val_end <= 1000
    assert val_end - train_end == 150


def test_difference_inverse_roundtrip():
    y = np.cumsum(np.random.default_rng(0).normal(size=(50, 2)), axis=0) + 10
    delta = difference(y, s=1)
    y0 = y[:1]
    recovered = inverse_difference(delta, y0, s=1)
    np.testing.assert_allclose(recovered, y, atol=1e-8)


def test_normalizer_fit_on_train_only():
    rng = np.random.default_rng(0)
    train = rng.normal(loc=5.0, scale=2.0, size=(100, 1))
    norm = TrainFittedNormalizer().fit(train)
    z = norm.transform(train)
    assert abs(z.mean()) < 1e-6
    assert abs(z.std() - 1.0) < 1e-6
    # inverse transform восстанавливает исходный масштаб
    np.testing.assert_allclose(norm.inverse_transform(z), train, atol=1e-6)


def test_prepare_and_window_no_leakage_in_shapes():
    series = generate_series(length=1000, n_features=1, seed=1)
    cfg = DataConfig(window_len=32, stride=1, n_features=1)
    prepared = prepare_and_window(series.values, cfg, regime=series.regime, is_anomaly=series.is_anomaly)
    assert prepared.train_windows.shape[1:] == (32, 1)
    assert prepared.val_windows.shape[0] > 0
    assert prepared.test_windows.shape[0] > 0
    assert not np.isnan(prepared.train_windows).any()


def test_all_models_forward_backward():
    cfg = DataConfig(window_len=24, n_features=1)
    x = torch.randn(4, cfg.window_len, cfg.n_features)
    for name in ALL_MODEL_NAMES:
        model_cfg = ModelConfig(model_name=name, latent_dim=8, hidden_dim=16,
                                 tcn_levels=3, d_model=16, n_heads=2, n_attn_layers=1)
        model = build_model(model_cfg, n_features=cfg.n_features, window_len=cfg.window_len)
        x_hat, z, mu, logvar = model(x)
        assert x_hat.shape == x.shape, name
        assert not torch.isnan(x_hat).any(), name
        loss = x_hat.mean()
        loss.backward()
        assert any(p.grad is not None for p in model.parameters()), name
