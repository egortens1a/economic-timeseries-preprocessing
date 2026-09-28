"""
Предобработка: детрендирование и Z-нормализация по обучающей выборке
"""
from dataclasses import dataclass

import numpy as np


def difference(y: np.ndarray, s: int = 1) -> np.ndarray:
    """deltay_t = y_t - y_{t-s}. Первые s значений отбрасываются"""
    if s <= 0:
        return y.copy()
    return y[s:] - y[:-s]


def inverse_difference(delta: np.ndarray, y0: np.ndarray, s: int = 1) -> np.ndarray:
    """Восстановление исходного уровня ряда по разностям и начальным s значениям"""
    if s <= 0:
        return delta.copy()
    length = delta.shape[0] + s
    out = np.zeros((length,) + delta.shape[1:], dtype=delta.dtype)
    out[:s] = y0
    for t in range(s, length):
        out[t] = out[t - s] + delta[t - s]
    return out


@dataclass
class TrainFittedNormalizer:
    """Z-нормализация со статистиками, зафиксированными на train-срезе"""

    eps: float = 1e-8
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None

    def fit(self, x_train: np.ndarray) -> "TrainFittedNormalizer":
        # x_train: (N, F)
        self.mean_ = x_train.mean(axis=0)
        self.std_ = x_train.std(axis=0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None and self.std_ is not None, "normalizer is not fitted"
        return (x - self.mean_) / (self.std_ + self.eps)

    def inverse_transform(self, x_norm: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None and self.std_ is not None, "normalizer is not fitted"
        return x_norm * (self.std_ + self.eps) + self.mean_


def prepare_series(
    raw: np.ndarray,
    seasonal_period: int = 1,
    train_frac: float = 0.70,
    log_transform: bool = False,
) -> dict:
    """
    Полный препроцессинг: 
    опциональное логарифмирование, детрендирование, разделение по времени,
    обучение нормализатора на train и нормализация всего ряда.
    
    log_transform=True рекомендуется для растущих ценовых рядов:
    дифференцирование log(raw) даёт масштабно-инвариантные log-return
    и предотвращает резкий рост val_loss. Требуются положительные значения
    """
    if log_transform:
        if np.any(raw <= 0):
            n_bad = int((raw <= 0).sum())
            raise ValueError(
                f"log_transform=True requires strictly positive values, found {n_bad} "
                "non-positive entries (e.g. zero-volume days). Clip/drop them first, "
                "or set log_transform=False."
            )
        raw_for_diff = np.log(raw)
    else:
        raw_for_diff = raw

    delta = difference(raw_for_diff, s=seasonal_period)
    n_train = int(len(delta) * train_frac)
    normalizer = TrainFittedNormalizer().fit(delta[:n_train])
    normalized = normalizer.transform(delta)
    return {
        "delta": delta,
        "normalized": normalized,
        "normalizer": normalizer,
        "y0": raw_for_diff[:seasonal_period],
        "n_train": n_train,
        "seasonal_period": seasonal_period,
        "log_transform": log_transform,
    }