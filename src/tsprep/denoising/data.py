"""Нормализация по обучающим данным и подготовка окон для каждой траектории."""
import numpy as np
from tsprep.data_preprocessing.datasets import sliding_windows
from tsprep.data_preprocessing.denoising import generate_denoising_series
from tsprep.data_preprocessing.preprocessing import TrainFittedNormalizer
from tsprep.provenance import input_fingerprint, save_npz, write_json


def series_arrays(series):
    return dict(observed=series.values, clean=series.clean_values, anomaly=series.is_anomaly)


def trajectory_segments(series, protocol, normalizer, *, development):
    observed = np.diff(np.log(series.values), axis=0)
    clean = np.diff(np.log(series.clean_values), axis=0)
    anomaly = series.is_anomaly[1:] | series.is_anomaly[:-1]
    split = int(len(observed)*protocol.train_fraction)
    bounds = {"train": (0, split), "val": (split, len(observed))} if development else {"test": (0, len(observed))}
    return {name: dict(observed=normalizer.transform(observed[lo:hi]).astype(np.float32),
                       clean=normalizer.transform(clean[lo:hi]).astype(np.float32),
                       anomaly=anomaly[lo:hi], start=lo)
            for name, (lo, hi) in bounds.items()}


def window_inputs(segment, p, stride=1, clean_input=False):
    x = segment["clean"] if clean_input else segment["observed"]
    return np.array(sliding_windows(x, p.window, stride), dtype=np.float32, copy=True)


def denoising_indices(segment, p, stride=1):
    """Return repaired endpoint and observation-availability indices, inside one segment."""
    availability = np.arange(p.window-1, len(segment["observed"]), stride)
    return availability-getattr(p, "correction_delay", 0), availability


def prepare_development(p, out):
    development = []
    raw_development = []
    (out/"inputs").mkdir()
    fingerprints = {}
    for family in p.families:
        for index, seed in enumerate(p.development_seeds):
            scales = getattr(p, "volatility_scales", (1.,))
            scale = scales[index % len(scales)]
            s = generate_denoising_series(family, seed, p.length, volatility_scale=scale)
            arrays = series_arrays(s)
            key = f"development_{family}_{seed}"
            save_npz(out/"inputs"/f"{key}.npz", arrays)
            fingerprints[key] = input_fingerprint(arrays, ["price"])
            val = None
            validation_seeds = getattr(p, "validation_seeds", ())
            if validation_seeds:
                val_seed = validation_seeds[index]
                val = generate_denoising_series(family, val_seed, p.length, volatility_scale=scale)
                key = f"validation_{family}_{val_seed}"
                arrays = series_arrays(val)
                save_npz(out/"inputs"/f"{key}.npz", arrays)
                fingerprints[key] = input_fingerprint(arrays, ["price"])
            
            raw_development.append((family, seed, s, val, scale, index))
    
    train_returns = [np.diff(np.log(s.values), axis=0)[:int((p.length-1)*p.train_fraction)] for _, _, s, _, _, _ in raw_development]
    normalizer = TrainFittedNormalizer().fit(np.concatenate(train_returns))
    write_json(out/"normalizer.json", normalizer.state())
    for family, seed, s, val, scale, index in raw_development:
        parts = trajectory_segments(s, p, normalizer, development=True)
        if val is not None:
            parts["val"] = trajectory_segments(val, p, normalizer, development=True)["val"]
        if not p.smoke and not all(v["anomaly"][denoising_indices(v, p)[0]].any() for v in parts.values()):
            raise ValueError("development segment has no contaminated endpoints")
        development.append(dict(family=family, seed=seed, volatility_scale=scale,
                                validation_seed=p.validation_seeds[index] if val is not None else seed, **parts))
    
    return development, normalizer, fingerprints


def representation_windows(segment, p, stride=1, severity=1):
    """Pair identical causal endpoints with clean targets strictly inside the segment."""
    if severity not in (0, 1, 2):
        raise ValueError("severity must be 0, 1 or 2")
    x = segment["observed"] if severity == 1 else segment["clean"] + severity*(segment["observed"]-segment["clean"])
    windows = np.array(sliding_windows(x, p.window, stride), np.float32, copy=True)
    ends = np.arange(p.window-1, len(x), stride)
    valid = ends+p.forecast_horizon < len(x)
    return windows[valid], ends[valid], segment["clean"][ends[valid]+p.forecast_horizon, 0]


def pooled_representation_data(development, split, p):
    stride = p.training_stride if split == "train" else 1
    observed, clean, targets, ends, groups = [], [], [], [], []
    offset = 0
    for item in development:
        segment = item[split]
        x, indices, y = representation_windows(segment, p, stride)
        observed.append(x)
        clean.append(window_inputs(segment, p, stride, clean_input=True)[:len(x)])
        targets.append(y)
        ends.append(indices+segment["start"])
        groups.append(dict(family=item["family"], data_seed=item["seed"], start=offset, stop=offset+len(x)))
        offset += len(x)
    return dict(windows=np.concatenate(observed), clean=np.concatenate(clean),
                target=np.concatenate(targets), ends=np.concatenate(ends), groups=groups)
