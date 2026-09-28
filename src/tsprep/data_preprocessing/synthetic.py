"""
Генератор синтетических "экономических" временных рядов.

- стохастический тренд (нестационарность первого порядка),
- многомасштабная сезонность,
- гетероскедастичность (GARCH-подобная волатильность),
- структурные сдвиги (переключение "рыночных режимов").

Генератор также возвращает скрытую метку режима (regime label) и
маску искусственно внесённых аномалий (выбросов) - они не участвуют
в обучении, а используются только на этапе оценки (downstream-задачи
кластеризации / обнаружения аномалий).
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class SyntheticSeries:
    values: np.ndarray        # (length, n_features)
    regime: np.ndarray        # (length,) int label of market regime
    is_anomaly: np.ndarray    # (length,) bool, injected point anomalies


def _garch_like_volatility(length: int, rng: np.random.Generator,
                            omega=1e-5, alpha=0.15, beta=0.80) -> np.ndarray:
    """GARCH(1,1)-подобная генерация условной дисперсии сигма_t^2."""
    sigma2 = np.zeros(length)
    eps = rng.standard_normal(length)
    sigma2[0] = omega / max(1e-8, (1 - alpha - beta))
    for t in range(1, length):
        sigma2[t] = omega + alpha * (eps[t - 1] ** 2) * sigma2[t - 1] + beta * sigma2[t - 1]
    return np.sqrt(np.maximum(sigma2, 1e-10))


def generate_series(
    length: int = 4000,
    n_features: int = 1,
    n_regimes: int = 3,
    anomaly_rate: float = 0.01,
    seed: int = 42,
) -> SyntheticSeries:
    """Одна синтетическая экономическая серия длиной ``length``."""
    rng = np.random.default_rng(seed)

    # 1) Скрытые режимы: случайные точки структурных сдвигов
    regime = np.zeros(length, dtype=int)
    n_breaks = max(1, n_regimes - 1)
    breakpoints = np.sort(rng.choice(np.arange(length // 10, length - length // 10),
                                      size=n_breaks, replace=False))
    cur = 0
    for i, bp in enumerate(breakpoints):
        regime[cur:bp] = i
        cur = bp
    regime[cur:] = n_breaks

    values = np.zeros((length, n_features))
    for f in range(n_features):
        # 2) Стохастический тренд (случайное блуждание со сменой сноса по режимам)
        drift_per_regime = rng.normal(0, 0.0008, size=n_regimes)
        drift = drift_per_regime[regime]

        # 3) Многомасштабная сезонность: недельная + квартальная гармоники
        t = np.arange(length)
        seasonal = (
            0.6 * np.sin(2 * np.pi * t / 5.0 + rng.uniform(0, 2 * np.pi))
            + 0.3 * np.sin(2 * np.pi * t / 63.0 + rng.uniform(0, 2 * np.pi))
        )

        # 4) Гетероскедастичный шум (GARCH-подобный)
        sigma = _garch_like_volatility(length, rng)
        # волатильность дополнительно скачет при смене режима
        regime_vol_mult = 1.0 + 0.5 * (regime / max(1, n_regimes - 1))
        noise = rng.standard_normal(length) * sigma * regime_vol_mult

        level = np.cumsum(drift) + seasonal * 0.05 + noise
        # экспонируем, чтобы получить положительный "ценовой" ряд
        price = 100 * np.exp(level - level[0])
        values[:, f] = price

    # 5) Точечные аномалии (внесённые скачки/провалы) - только для оценки
    is_anomaly = rng.random(length) < anomaly_rate
    anomaly_idx = np.where(is_anomaly)[0]
    for idx in anomaly_idx:
        shock = rng.choice([-1, 1]) * rng.uniform(4, 8)
        values[idx, :] *= (1 + shock * 0.01 * rng.uniform(0.5, 1.5))

    return SyntheticSeries(values=values, regime=regime, is_anomaly=is_anomaly)


def generate_dataset(
    n_series: int = 8,
    length: int = 4000,
    n_features: int = 1,
    seed: int = 42,
) -> list[SyntheticSeries]:
    """Набор из нескольких независимых синтетических рядов (разные seed)."""
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 1_000_000, size=n_series)
    return [generate_series(length=length, n_features=n_features, seed=int(s)) for s in seeds]
