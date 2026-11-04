"""Train-fitted transformations with a serializable inverse."""
from dataclasses import dataclass
import numpy as np


def difference(y, s=1):
    if not isinstance(s, int) or s < 0 or s >= len(y):
        raise ValueError("difference period must be in [0, len(y))")
    return y.copy() if s == 0 else y[s:] - y[:-s]


def inverse_difference(delta, y0, s=1):
    if s < 0:
        raise ValueError("negative period")
    if s == 0:
        return delta.copy()
    if len(y0) != s:
        raise ValueError("inverse difference requires exactly s initial observations")
    out = np.empty((len(delta)+s,) + delta.shape[1:], dtype=float)
    out[:s] = y0
    for t in range(s, len(out)):
        out[t] = out[t-s] + delta[t-s]
    return out


@dataclass
class TrainFittedNormalizer:
    eps: float = 1e-8
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None

    def fit(self, x_train):
        if x_train.ndim != 2 or len(x_train) < 2 or not np.isfinite(x_train).all():
            raise ValueError("normalizer requires at least two finite (N,F) samples")
        self.mean_, self.std_ = x_train.mean(axis=0), x_train.std(axis=0)
        return self

    def _check(self, x):
        if self.mean_ is None or self.std_ is None:
            raise ValueError("normalizer is not fitted")
        if x.shape[-1] != len(self.mean_) or not np.isfinite(x).all():
            raise ValueError("invalid features or nonfinite input")

    def transform(self, x):
        self._check(x)
        return (x-self.mean_) / (self.std_+self.eps)

    def inverse_transform(self, x):
        self._check(x)
        return x*(self.std_+self.eps)+self.mean_

    def state(self):
        self._check(self.mean_[None] if self.mean_ is not None else np.empty((0, 0)))
        return dict(eps=self.eps, mean=self.mean_.tolist(), std=self.std_.tolist())

    @classmethod
    def from_state(cls, state):
        obj = cls(eps=float(state["eps"]), mean_=np.asarray(state["mean"], dtype=float),
                  std_=np.asarray(state["std"], dtype=float))
        if obj.eps <= 0 or not np.isfinite(obj.eps) or obj.mean_.ndim != 1 or obj.mean_.shape != obj.std_.shape:
            raise ValueError("invalid normalizer state")
        if not np.isfinite(obj.mean_).all() or not np.isfinite(obj.std_).all() or (obj.std_ < 0).any():
            raise ValueError("invalid normalizer state")
        return obj


def prepare_series(raw, seasonal_period=1, train_frac=.70, log_transform=True, normalizer=None):
    raw = np.asarray(raw, dtype=float)
    if raw.ndim != 2 or not np.isfinite(raw).all() or raw.shape[1] == 0:
        raise ValueError("raw must be a finite (N,F) array")
    if not 0 < train_frac < 1:
        raise ValueError("invalid train_frac")
    if log_transform and np.any(raw <= 0):
        raise ValueError("log_transform requires positive features; choose columns/transforms explicitly")
    transformed = np.log(raw) if log_transform else raw.copy()
    delta = difference(transformed, seasonal_period)
    n_train = int(len(delta)*train_frac)
    if normalizer is None:
        normalizer = TrainFittedNormalizer().fit(delta[:n_train])
    return dict(delta=delta, normalized=normalizer.transform(delta), normalizer=normalizer,
                y0=transformed[:seasonal_period].copy(), n_train=n_train,
                seasonal_period=seasonal_period, log_transform=log_transform)


def inverse_series(normalized, normalizer, y0, seasonal_period=1, log_transform=True):
    levels = inverse_difference(normalizer.inverse_transform(normalized), y0, seasonal_period)
    return np.exp(levels) if log_transform else levels
