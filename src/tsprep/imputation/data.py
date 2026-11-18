"""Загрузка панели рядов, нормализация и дополнительные пропуски для оценки."""
from pathlib import Path
import zipfile

from tsprep.config import ImputationProtocol
import numpy as np





def load_panel(path):
    """Read a values[series, time] NPZ or the original daily NN5 TSF ZIP."""
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            members = [name for name in archive.namelist() if name.lower().endswith(".tsf")]
            if len(members) != 1:
                raise ValueError("expected exactly one TSF file in the archive")
            text = archive.read(members[0]).decode("cp1252")
        if "@frequency daily" not in text:
            raise ValueError("TSF input must have daily frequency")
     
        records = [line.strip().split(":", 2) for line in text.splitlines()
                   if line.strip() and not line.strip().startswith(("#", "@"))]
        if not records or any(len(row) != 3 for row in records):
            raise ValueError("expected TSF rows: series_name:start_timestamp:values")
       
        starts = [np.datetime64(row[1].split()[0], "D") for row in records]
        rows = [np.array([float(v) if v != "?" else np.nan for v in row[2].split(",")])
                for row in records]
        if len(set(starts)) != 1 or len({len(row) for row in rows}) != 1:
            raise ValueError("series must have a common start and length")
      
        panel = dict(values=np.stack(rows), series_ids=np.array([row[0] for row in records]),
                     dates=(starts[0]+np.arange(len(rows[0]))).astype(str))
    else:
        with np.load(path, allow_pickle=False) as archive:
            panel = {key: archive[key] for key in archive.files}
    
    values = np.asarray(panel.get("values"), dtype=float)
    if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
        raise ValueError("values must be a nonempty [series, time] matrix")
   
    if np.isinf(values).any() or (values[np.isfinite(values)] < 0).any():
        raise ValueError("observations must be nonnegative and finite; use NaN for missing cells")
   
    panel["values"] = values
    panel["missing"] = np.isnan(values)
    ids = np.asarray(panel.get("series_ids", np.arange(len(values)).astype(str))).astype(str)
    if ids.shape != (len(values),) or len(set(ids)) != len(ids):
        raise ValueError("series_ids must be unique with one entry per row")
 
    panel["series_ids"] = ids
    if "dates" in panel:
        dates = np.asarray(panel["dates"], dtype="datetime64[D]")
        if dates.shape != (values.shape[1],) or np.isnat(dates).any() or not (np.diff(dates) == np.timedelta64(1, "D")).all():
            raise ValueError("dates must contain consecutive daily dates, one per column")
        panel["dates"] = dates.astype(str)
  
    return panel


def fit_scaler(raw, p=ImputationProtocol()):
    if any(not np.isfinite(raw[:, day:p.train_stop:p.period]).any(axis=1).all() for day in range(p.period)):
        raise ValueError("every series needs a known training observation in each seasonal phase")
  
    log = np.log1p(raw)
    mean = np.nanmean(log[:, :p.train_stop], axis=1)
    std = np.maximum(np.nanstd(log[:, :p.train_stop], axis=1), 1e-6)
    seasonal_error = np.abs(raw[:, p.period:p.train_stop]-raw[:, :p.train_stop-p.period])
    if not np.isfinite(seasonal_error).any(axis=1).all():
        raise ValueError("every series needs a known training pair one seasonal period apart")
 
    mase = np.maximum(np.nanmean(seasonal_error, axis=1), 1e-6)
    normalized = (log-mean[:, None])/std[:, None]
    profile = np.stack([np.nanmedian(normalized[:, day:p.train_stop:p.period], axis=1) for day in range(p.period)], axis=1)
    lengths = []
    for missing in np.isnan(raw[:, :p.train_stop]):
        edges = np.flatnonzero(np.diff(np.r_[False, missing, False].astype(int)))
        lengths.extend(int(b-a) for a, b in zip(edges[::2], edges[1::2]))
  
    support, counts = np.unique(lengths or [1], return_counts=True)
    return dict(mean=mean, std=std, mase=mase, profile=profile,
                gap_lengths=support, gap_probabilities=counts/counts.sum()), normalized


def sample_mask(values, rng, scaler, fraction=.05):
    """Exact per-row target count; empirical block lengths; never score original NaNs."""
    known = np.isfinite(values)
    target = np.ceil(fraction*known.sum(axis=1)).astype(int)
    hidden = np.zeros(values.shape, bool)
    count = np.zeros(len(values), int)
    for _ in range(1000):
        rows = np.flatnonzero(count < target)
        if not len(rows):
            return hidden
   
        starts = rng.integers(0, values.shape[1], len(rows))
        lengths = rng.choice(scaler["gap_lengths"], len(rows), p=scaler["gap_probabilities"])
        for offset in range(int(lengths.max())):
            positions = starts+offset
            valid = (offset < lengths) & (positions < values.shape[1]) & (count[rows] < target[rows])
            rr, cc = rows[valid], positions[valid]
            good = known[rr, cc] & ~hidden[rr, cc]
            hidden[rr[good], cc[good]] = True
            count[rr[good]] += 1
  
    raise RuntimeError("Could not construct the prescribed missing mask")


def additional_masks(normalized, scaler, lo, hi, seeds, fraction=.05):
    masks = []
    for seed in seeds:
        extra = np.zeros(normalized.shape, bool)
        extra[:, lo:hi] = sample_mask(normalized[:, lo:hi], np.random.default_rng(seed), scaler, fraction)
        masks.append(extra)
  
    return np.stack(masks)


def train_windows(normalized, p=ImputationProtocol()):
    starts = np.arange(0, p.train_stop-p.window+1, p.period)
    atm = np.repeat(np.arange(len(normalized)), len(starts))
    start = np.tile(starts, len(normalized))
    values = np.stack([normalized[s, a:a+p.window] for s, a in zip(atm, start)])
    return dict(windows=values, atm=atm, start=start)


def evaluation_windows(raw, normalized, extra_masks, lo, hi, p=ImputationProtocol()):
    assert (hi-lo) % p.period == 0
    weeks = (hi-lo)//p.period
    atm = np.repeat(np.arange(len(raw)), weeks)
    starts = np.tile(np.arange(lo+p.period-p.window, hi-p.window+1, p.period), len(raw))
    truth = np.stack([normalized[s, a:a+p.window] for s, a in zip(atm, starts)])
    truth_raw = np.stack([raw[s, a:a+p.window] for s, a in zip(atm, starts)])
    values, scores = [], []
    for extra in extra_masks:
        hidden = np.stack([extra[s, a:a+p.window] for s, a in zip(atm, starts)])
        masked = truth.copy()
        masked[hidden] = np.nan
        score = hidden.copy()
        score[:, :-p.period] = False
        assert np.isfinite(truth_raw[score]).all()
        values.append(masked)
        scores.append(score)
  
    return dict(windows=np.concatenate(values), score_mask=np.concatenate(scores),
                truth=np.tile(truth, (len(extra_masks), 1)), truth_raw=np.tile(truth_raw, (len(extra_masks), 1)),
                atm=np.tile(atm, len(extra_masks)), start=np.tile(starts, len(extra_masks)),
                draws=len(extra_masks), weeks=weeks, series=len(raw))


def calendar_profile(atm, starts, scaler, window=56):
    days = (np.asarray(starts)[:, None]+np.arange(window)) % scaler["profile"].shape[1]
    profile = scaler["profile"][np.asarray(atm)[:, None], days]
    return days, profile


def clip_missing(prediction, windows, atm, scaler):
    lower = -scaler["mean"][atm]/scaler["std"][atm]
    return np.where(np.isfinite(windows), windows, np.maximum(prediction, lower[:, None]))


def native_values(prediction, data, scaler):
    estimated = np.maximum(np.expm1(prediction*scaler["std"][data["atm"], None]+scaler["mean"][data["atm"], None]), 0.)
    # Копируем доступные значения. Скрытый эталон не используется при заполнении.
    return np.where(np.isfinite(data["windows"]), data["truth_raw"], estimated)


def error_blocks(prediction, data, scaler):
    known = np.isfinite(data["windows"])
    np.testing.assert_array_equal(prediction[known], data["windows"][known])
    raw = native_values(prediction, data, scaler)
    if not np.isfinite(raw).all():
        raise FloatingPointError("nonfinite reconstructed values")
  
    if not data["score_mask"].any():
        raise ValueError("no known observations were hidden for evaluation")
  
    error = np.where(data["score_mask"], np.abs(raw-data["truth_raw"]), 0.)
    squared = np.where(data["score_mask"], (raw-data["truth_raw"])**2, 0.)
    scaled = error/scaler["mase"][data["atm"], None]
    shape = data["draws"], data["series"], data["weeks"], data["windows"].shape[1]
    return dict(scaled=scaled.reshape(shape).sum(axis=-1), absolute=error.reshape(shape).sum(axis=-1),
                squared=squared.reshape(shape).sum(axis=-1),
                count=data["score_mask"].reshape(shape).sum(axis=-1),
                score=float(scaled.sum()/data["score_mask"].sum()),
                native_mae=float(error.sum()/data["score_mask"].sum()))


def forecast_windows(raw, normalized, lo, hi, extra=None, p=ImputationProtocol()):
    targets = np.arange(max(lo, p.window), hi)
    atm = np.repeat(np.arange(len(raw)), len(targets))
    target_index = np.tile(targets, len(raw))
    keep = np.isfinite(raw[atm, target_index])
    atm, target_index = atm[keep], target_index[keep]
    source = normalized.copy()
    if extra is not None:
        source[extra] = np.nan
   
    start = target_index-p.window
    windows = np.stack([source[s, a:a+p.window] for s, a in zip(atm, start)])
    return dict(windows=windows, atm=atm, start=start, target_index=target_index, period=p.period,
                target_normalized=normalized[atm, target_index], target=raw[atm, target_index])
