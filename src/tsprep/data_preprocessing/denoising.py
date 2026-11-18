"""Генерация чистых рядов и добавление ошибок наблюдения."""
import numpy as np

from tsprep.data_preprocessing.synthetic import SyntheticSeries


FAMILIES = (
    "garch_random_walk_no_seasonality", "ar1_returns_student_t",
    "stochastic_volatility", "seasonal_period_11",
)


def generate_denoising_series(family, seed, length=6000, *, contaminated=True,
                              measurement_std=.001, impulse_probability=.01,
                              impulse_bounds=(.04, .08), impulse_duration=1, volatility_scale=1.):
    """Четыре семейства рядов. Динамика и ошибки наблюдения генерируются раздельно.

    В clean остаются случайные изменения сигнала. Ошибка одной цены влияет
    на обе соседние логарифмические разности.
    """
    if family not in FAMILIES or length < 100:
        raise ValueError("unknown family or trajectory shorter than 100")
    if not np.isfinite(volatility_scale) or volatility_scale <= 0:
        raise ValueError("volatility scale must be finite and positive")
    if (not np.isfinite(measurement_std) or measurement_std < 0 or
        not 0 <= impulse_probability <= 1 or len(impulse_bounds) != 2 or
        not np.isfinite(impulse_bounds).all() or not 0 <= impulse_bounds[0] <= impulse_bounds[1] or
        not isinstance(impulse_duration, int) or not 1 <= impulse_duration <= length):
        raise ValueError("invalid observation error parameters")
    streams = np.random.SeedSequence([seed, FAMILIES.index(family)]).spawn(4)
    economic, phases, measurement, impulses = [np.random.default_rng(s) for s in streams]
    burn = 500
    returns = np.empty(length + burn)
    if family == "ar1_returns_student_t":
        previous = 0.
        for t in range(len(returns)):
            previous = .6 * previous + .004 * np.sqrt(3/5) * economic.standard_t(5)
            returns[t] = previous
    elif family == "stochastic_volatility":
        m = np.log(.00002)
        log_variance = m
        for t in range(len(returns)):
            log_variance = m + .98 * (log_variance-m) + .15 * economic.normal()
            returns[t] = np.exp(log_variance/2) * economic.normal()
    else:
        variance = .00002
        for t in range(len(returns)):
            returns[t] = np.sqrt(variance) * economic.normal()
            variance = .000001 + .1 * returns[t]**2 + .85 * variance
    level = np.cumsum(returns[burn:])
    if family == "seasonal_period_11":
        t = np.arange(length)
        level += .015 * np.sin(2*np.pi*t/11 + phases.uniform(0, 2*np.pi))
        level += .0075 * np.sin(2*np.pi*t/67 + phases.uniform(0, 2*np.pi))
    # Scale retained dynamics, including seasonal components; observation errors stay fixed.
    clean = (100 * np.exp(volatility_scale*(level-level[0])))[:, None]
    noise = measurement.normal(0, measurement_std, size=clean.shape)
    anomaly = impulses.random(length) < impulse_probability
    shock = impulses.choice([-1, 1], length) * impulses.uniform(*impulse_bounds, length)
    if impulse_duration > 1:
        # Each event is a persistent price-level error; overlapping events add.
        shock = np.convolve(anomaly*shock, np.ones(impulse_duration), mode="full")[:length]
        anomaly = np.convolve(anomaly.astype(int), np.ones(impulse_duration, int), mode="full")[:length] > 0
    if not contaminated:
        noise[:] = 0
        anomaly[:] = False
    observed = clean * np.exp(noise + (anomaly*shock)[:, None])
    if not np.isfinite(observed).all() or (observed <= 0).any():
        raise FloatingPointError("nonfinite or nonpositive generated prices")
    return SyntheticSeries(observed, np.zeros(length, int), anomaly, clean, noise)
