"""Chronological windows, exact forecast indices, and preprocessing metadata."""
from dataclasses import dataclass, field
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tsprep.data_preprocessing.preprocessing import prepare_series


def sliding_windows(x, window_len, stride=1):
    if not isinstance(window_len, int) or not isinstance(stride, int) or window_len < 1 or stride < 1:
        raise ValueError("window_len and stride must be positive integers")
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or len(x) < window_len:
        raise ValueError(f"Series too short or invalid shape {x.shape} for window_len={window_len}")
    # A read-only view avoids materializing T copies of every point.
    return np.moveaxis(np.lib.stride_tricks.sliding_window_view(x, window_len, axis=0), -1, 1)[::stride]


def chronological_split_indices(n, train_frac, val_frac, test_frac):
    fractions = (train_frac, val_frac, test_frac)
    if n < 1 or any(not 0 < x < 1 for x in fractions) or not np.isclose(sum(fractions), 1., atol=1e-8, rtol=0):
        raise ValueError("invalid chronological split")
    train_end = int(n*train_frac)
    return train_end, train_end+int(n*val_frac)


class WindowDataset(Dataset):
    def __init__(self, windows, targets=None):
        if windows.ndim != 3 or not len(windows):
            raise ValueError("windows must be nonempty (N,T,F)")
        self.windows = windows
        if targets is not None and targets.shape != windows.shape:
            raise ValueError("targets must match input windows")
        self.targets = targets

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        x = torch.tensor(self.windows[index], dtype=torch.float32)
        return x if self.targets is None else (x, torch.tensor(self.targets[index], dtype=torch.float32))


@dataclass
class PreparedData:
    train_windows: np.ndarray
    val_windows: np.ndarray
    test_windows: np.ndarray
    normalizer: object
    regime: np.ndarray | None = None
    is_anomaly: np.ndarray | None = None
    raw_normalized: np.ndarray | None = None
    window_starts: dict = field(default_factory=dict)
    split_bounds: dict = field(default_factory=dict)
    dates: np.ndarray | None = None
    y0: np.ndarray | None = None
    feature_names: list = field(default_factory=list)
    seasonal_period: int = 1
    log_transform: bool = True


def prepare_and_window(raw, cfg, regime=None, is_anomaly=None, *, dates=None, feature_names=None, normalizer=None):
    cfg.__post_init__()
    raw = np.asarray(raw, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != cfg.n_features:
        raise ValueError("raw shape does not match n_features")
    for name, values in (("regime", regime), ("is_anomaly", is_anomaly), ("dates", dates)):
        if values is not None and (np.asarray(values).ndim != 1 or len(values) != len(raw)):
            raise ValueError(f"{name} must have one value per raw observation")
    prep = prepare_series(raw, cfg.seasonal_period, cfg.train_frac, cfg.log_transform, normalizer)
    x = prep["normalized"]
    train_end, val_end = chronological_split_indices(len(x), cfg.train_frac, cfg.val_frac, cfg.test_frac)
    bounds = dict(train=(0, train_end), val=(train_end, val_end), test=(val_end, len(x)))
    windows, starts = {}, {}
    for name, (lo, hi) in bounds.items():
        if hi-lo < cfg.window_len + cfg.forecast_horizon:
            raise ValueError(f"{name} needs at least window_len + horizon observations")
        windows[name] = sliding_windows(x[lo:hi], cfg.window_len, cfg.stride)
        starts[name] = np.arange(lo, hi-cfg.window_len+1, cfg.stride)
    s = cfg.seasonal_period
    mask = None
    if is_anomaly is not None:
        a = np.asarray(is_anomaly, dtype=bool)
        mask = (a[s:] | a[:-s]) if s else a.copy()
    names = list(feature_names) if feature_names is not None else [f"feature_{i}" for i in range(cfg.n_features)]
    if len(names) != cfg.n_features or len(set(names)) != len(names):
        raise ValueError("feature_names must be unique and match n_features")
    return PreparedData(windows["train"], windows["val"], windows["test"], prep["normalizer"],
        None if regime is None else np.asarray(regime)[s:], mask, x, starts, bounds,
        None if dates is None else np.asarray(dates)[s:], prep["y0"], names, s, cfg.log_transform)


def forecast_pairs(prepared, split, cfg):
    ends = prepared.window_starts[split] + cfg.window_len - 1
    targets = ends + cfg.forecast_horizon
    valid = targets < prepared.split_bounds[split][1]
    return np.flatnonzero(valid), prepared.raw_normalized[targets[valid], cfg.target_feature]


def build_dataloaders(prepared, cfg, *, train_targets=None, val_targets=None):
    generator = torch.Generator().manual_seed(cfg.seed)
    train = DataLoader(WindowDataset(prepared.train_windows, train_targets), batch_size=cfg.batch_size,
                       shuffle=True, drop_last=False, generator=generator)
    return train, build_eval_loader(prepared.val_windows, cfg.batch_size, targets=val_targets), build_eval_loader(prepared.test_windows, cfg.batch_size)


def build_eval_loader(windows, batch_size, *, targets=None):
    return DataLoader(WindowDataset(windows, targets), batch_size=batch_size, shuffle=False, drop_last=False)


def aggregate_windows(windows, starts, length, *, mode="last"):
    """Map reconstructions to time. 'last' is causal at the window end; 'mean' is offline."""
    if mode not in ("last", "mean") or len(starts) != len(windows):
        raise ValueError("invalid aggregation")
    sums = np.zeros((length, windows.shape[2]))
    counts = np.zeros(length)
    for window, start in zip(windows, starts):
        if start < 0 or start+len(window) > length:
            raise ValueError("window outside target range")
        indices = np.array([start+len(window)-1]) if mode == "last" else np.arange(start, start+len(window))
        sums[indices] += window[-1:] if mode == "last" else window
        counts[indices] += 1
    out = np.full_like(sums, np.nan)
    covered = counts > 0
    out[covered] = sums[covered] / counts[covered, None]
    return out, covered
