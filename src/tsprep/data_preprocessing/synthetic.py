"""Recurrent regimes, random walk, GARCH innovations and explicit clean/noisy pairs."""
from dataclasses import dataclass
import numpy as np


@dataclass
class SyntheticSeries:
    values: np.ndarray
    regime: np.ndarray
    is_anomaly: np.ndarray
    clean_values: np.ndarray
    observation_noise: np.ndarray


def _garch_innovations(length, rng, omega=1e-6, alpha=.15, beta=.80):
    if omega <= 0 or min(alpha, beta) < 0 or alpha+beta >= 1:
        raise ValueError("GARCH parameters must define a stationary positive variance")
    variance = omega / (1-alpha-beta)
    innovations = np.empty(length)
    for t in range(length):
        innovations[t] = np.sqrt(variance)*rng.standard_normal()
        variance = omega + alpha*innovations[t]**2 + beta*variance
    return innovations


def generate_series(length=4000, n_features=1, n_regimes=3, anomaly_rate=.01, seed=42):
    if length < 10 or n_features < 1 or n_regimes < 1 or not 0 <= anomaly_rate < 1:
        raise ValueError("invalid synthetic series parameters")
    rng = np.random.default_rng(seed)
    regime = np.zeros(length, dtype=int)
    # Recurring regimes, with a minimum dwell time; no dependence on the split.
    current, start = int(rng.integers(n_regimes)), 0
    while start < length:
        stop = min(length, start+int(rng.integers(60, 140)))
        regime[start:stop] = current
        if n_regimes > 1:
            current = (current+int(rng.integers(1, n_regimes))) % n_regimes
        start = stop
    t = np.arange(length)
    clean = np.empty((length, n_features))
    for f in range(n_features):
        drift = np.linspace(-.0008, .0008, n_regimes)[regime]
        innovations = _garch_innovations(length, rng)
        # Regime modulation is explicit, separate from base GARCH.
        level = np.cumsum(drift + innovations*(1+.5*regime))
        seasonal = .03*np.sin(2*np.pi*t/5 + rng.uniform(0, 2*np.pi))
        seasonal += .015*np.sin(2*np.pi*t/63 + rng.uniform(0, 2*np.pi))
        level += seasonal
        clean[:, f] = 100*np.exp(level-level[0])
    observation_noise = rng.normal(0, .001, size=clean.shape)
    values = clean*np.exp(observation_noise)
    anomaly = rng.random(length) < anomaly_rate
    shocks = rng.choice([-1, 1], size=int(anomaly.sum()))*rng.uniform(.04, .08, size=int(anomaly.sum()))
    values[anomaly] *= np.exp(shocks[:, None])
    return SyntheticSeries(values, regime, anomaly, clean, observation_noise)


def generate_dataset(n_series=8, length=4000, n_features=1, seed=42):
    if n_series < 1:
        raise ValueError("n_series must be positive")
    rng = np.random.default_rng(seed)
    return [generate_series(length, n_features, seed=int(s)) for s in rng.integers(0, 1000000, n_series)]
