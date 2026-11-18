"""Методы заполнения пропусков; известные значения сохраняются без изменений."""
import warnings
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from .data import calendar_profile, clip_missing, sample_mask


METHODS = ("ffill", "weekday_train", "weekday_window", "linear", "pca", "ridge_gap")
ALPHAS = (.1, 1., 10., 100.)


def contexts(windows, atm, start, scaler):
    period = scaler["profile"].shape[1]
    known = np.isfinite(windows)
    days, profile = calendar_profile(atm, start, scaler, windows.shape[1])
    n = windows.shape[1]
    positions = np.arange(n)[None, :]
    left = np.maximum.accumulate(np.where(known, positions, -1), axis=1)
    right = np.minimum.accumulate(np.where(known, positions, n)[:, ::-1], axis=1)[:, ::-1]
    finite = np.nan_to_num(windows)
    lv = np.take_along_axis(finite, np.clip(left, 0, n-1), axis=1)
    rv = np.take_along_axis(finite, np.clip(right, 0, n-1), axis=1)
    linear = lv+(rv-lv)*(positions-left)/np.maximum(right-left, 1)
    linear = np.where(left < 0, rv, np.where(right >= n, lv, linear))
    linear = np.where((left < 0) & (right >= n), profile, linear)
    forward = np.where(left >= 0, lv, profile)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # Если в окне нет данных для дня недели, берём медиану обучающей выборки.
        weekday = np.tile(np.nanmedian(windows.reshape(-1, n//period, period), axis=1), (1, n//period))
   
    weekday = np.where(np.isfinite(weekday), weekday, profile)
    return dict(known=known, days=days, profile=profile, linear=linear, ffill=forward,
                weekday_window=weekday, left=left, right=right, lv=lv, rv=rv)


def simple(method, data, scaler, context=None):
    x = data["windows"]
    c = contexts(x, data["atm"], data["start"], scaler) if context is None else context
    value = c["profile"] if method == "weekday_train" else c[method]
    return clip_missing(value, x, data["atm"], scaler)


def pca_fill(data, scaler, state, iterations=20):
    x = data["windows"]
    value = simple("linear", data, scaler)
    components, mean = state["components"], state["mean"]
    for _ in range(iterations):
        reconstructed = ((value-mean) @ components.T) @ components+mean
        value = clip_missing(reconstructed, x, data["atm"], scaler)
   
    return value


def fit_pca(train, scaler, rank):
    value = simple("linear", train, scaler)
    for _ in range(5):
        pca = PCA(rank, svd_solver="full").fit(value)
        reconstructed = pca.inverse_transform(pca.transform(value))
        value = clip_missing(reconstructed, train["windows"], train["atm"], scaler)
  
    return dict(mean=pca.mean_, components=pca.components_, rank=rank)


def gap_features(data, scaler, rows, columns, context=None):
    x = data["windows"]
    c = contexts(x, data["atm"], data["start"], scaler) if context is None else context
    base = c["linear"][rows, columns]
    values = [base, c["profile"][rows, columns]-base, c["weekday_window"][rows, columns]-base]
    window, period = x.shape[1], scaler["profile"].shape[1]
    for offset in (-2*period, -period, -2, -1, 1, 2, period, 2*period):
        position = columns+offset
        within = (position >= 0) & (position < window)
        position = np.clip(position, 0, window-1)
        available = within & c["known"][rows, position]
        observed = np.where(available, np.nan_to_num(x[rows, position]), c["profile"][rows, position])
        values.extend([observed-base, available.astype(float)])
   
    for side in ("left", "right"):
        position = c[side][rows, columns]
        available = (position >= 0) & (position < window)
        distance = np.where(available, np.abs(position-columns)/window, 1.)
        observed = c["lv" if side == "left" else "rv"][rows, columns]
        values.extend([distance, np.where(available, observed, base)-base])
   
    phase = 2*np.pi*c["days"][rows, columns]/period
    values.extend([np.sin(phase), np.cos(phase)])
    features = np.column_stack(values)
    assert np.isfinite(features).all()
    return features, base


def ridge_fill(data, scaler, state):
    value = simple("linear", data, scaler)
    rows, columns = np.nonzero(~np.isfinite(data["windows"]))
    if len(rows):
        features, base = gap_features(data, scaler, rows, columns)
        value[rows, columns] = base+((features-state["mean"])/state["std"]) @ state["coef"]+state["intercept"]
  
    return clip_missing(value, data["windows"], data["atm"], scaler)


def fit_ridge(train, scaler, alpha, fraction=.05):
    features, targets = [], []
    for seed in (21001, 21002, 21003):
        extra = sample_mask(train["windows"], np.random.default_rng(seed), scaler, fraction)
        masked = train["windows"].copy()
        masked[extra] = np.nan
        case = dict(train, windows=masked)
        rows, columns = np.nonzero(extra)
        x, base = gap_features(case, scaler, rows, columns)
        features.append(x)
        targets.append(train["windows"][rows, columns]-base)
    
    x, y = np.concatenate(features), np.concatenate(targets)
    mean, std = x.mean(axis=0), x.std(axis=0)+1e-8
    ridge = Ridge(alpha=alpha).fit((x-mean)/std, y)
    return dict(mean=mean, std=std, coef=ridge.coef_, intercept=float(ridge.intercept_), alpha=alpha)


def predict(method, data, scaler, fitted):
    if method == "pca":
        return pca_fill(data, scaler, fitted["pca"])
   
    if method == "ridge_gap":
        return ridge_fill(data, scaler, fitted["ridge_gap"])
  
    return simple(method, data, scaler)


def network_features(data, scaler, base):
    context = contexts(data["windows"], data["atm"], data["start"], scaler)
    days, profile = context["days"], context["weekday_window"]
    angle = 2*np.pi*days/scaler["profile"].shape[1]
    return np.stack([base, np.isfinite(data["windows"]), profile, np.sin(angle), np.cos(angle)], axis=-1).astype(np.float32)
