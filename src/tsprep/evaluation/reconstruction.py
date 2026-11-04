"""
Метрики реконструкции и извлечение эмбеддингов
"""
import numpy as np
import torch
from torch.utils.data import DataLoader

from tsprep.models.autoencoder import Autoencoder


def mse(x: np.ndarray, x_hat: np.ndarray) -> float:
    return float(np.mean((x - x_hat) ** 2))


def mae(x: np.ndarray, x_hat: np.ndarray) -> float:
    return float(np.mean(np.abs(x - x_hat)))


def r2_score(x: np.ndarray, x_hat: np.ndarray) -> float:
    """R^2 = 1 - SS_res / SS_tot, посчитанный по всем точкам (B*T*F)."""
    x_flat = x.reshape(-1)
    xh_flat = x_hat.reshape(-1)
    ss_res = np.sum((x_flat - xh_flat) ** 2)
    ss_tot = np.sum((x_flat - x_flat.mean()) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-12))


@torch.no_grad()
def collect_embeddings(model: Autoencoder, loader: DataLoader, device: str = "cpu") -> dict:
    """Собирает z, входы, реконструкции и KL в eval-режиме; для VAE z=mu."""
    model.eval()
    model.to(device)
    zs, xs, xhs, kls = [], [], [], []
    for batch in loader:
        x = batch.to(device)
        x_hat, z, mu, logvar = model(x)
        z_out = mu if model.variational else z
        zs.append(z_out.cpu().numpy())
        xs.append(x.cpu().numpy())
        xhs.append(x_hat.cpu().numpy())
        if logvar is not None:
            kls.append((-0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(dim=1)).cpu().numpy())
    if not xs:
        raise ValueError("Cannot collect embeddings from an empty loader")
    return {
        "z": np.concatenate(zs, axis=0),
        "x": np.concatenate(xs, axis=0),
        "x_hat": np.concatenate(xhs, axis=0),
        "kl": np.concatenate(kls) if kls else np.zeros(sum(len(x) for x in xs)),
    }


def reconstruction_report(x: np.ndarray, x_hat: np.ndarray) -> dict:
    if x.shape != x_hat.shape or x.ndim != 3 or not np.isfinite(x).all() or not np.isfinite(x_hat).all():
        raise ValueError("Reconstruction arrays must be finite, aligned (N,T,F)")
    return {"mse": mse(x, x_hat), "mae": mae(x, x_hat), "r2": r2_score(x, x_hat),
            "per_feature": [{"mse": mse(x[..., i], x_hat[..., i]),
                             "mae": mae(x[..., i], x_hat[..., i]),
                             "r2": r2_score(x[..., i], x_hat[..., i])} for i in range(x.shape[-1])]}
