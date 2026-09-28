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
    """
    Прогоняет модель в eval-режиме по всему loaderу и собирает:
      - z: латентные представления (mu, если VAE - детерминированно)
      - x, x_hat: исходные и восстановленные окна (для метрик реконструкции)
    """
    model.eval()
    model.to(device)
    zs, xs, xhs = [], [], []
    for batch in loader:
        x = batch.to(device)
        x_hat, z, mu, logvar = model(x)
        z_out = mu if model.variational else z
        zs.append(z_out.cpu().numpy())
        xs.append(x.cpu().numpy())
        xhs.append(x_hat.cpu().numpy())
    return {
        "z": np.concatenate(zs, axis=0),
        "x": np.concatenate(xs, axis=0),
        "x_hat": np.concatenate(xhs, axis=0),
    }


def reconstruction_report(x: np.ndarray, x_hat: np.ndarray) -> dict:
    return {"mse": mse(x, x_hat), "mae": mae(x, x_hat), "r2": r2_score(x, x_hat)}