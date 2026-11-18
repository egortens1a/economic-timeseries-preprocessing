"""Ошибки заполнения, парные интервалы и отдельная проверка прогноза Ridge."""
import numpy as np
from sklearn.linear_model import Ridge

from tsprep.config import ImputationProtocol


ALPHAS = (.1, 1., 10., 100.)


def forecast_features(filled, data):
    period = data.get("period", 7)
    weekday = (data["target_index"] % period).astype(int)
    return np.c_[filled, ~np.isfinite(data["windows"]), np.eye(period)[weekday]]


def forecast_predict(probe, x, data, scaler):
    normalized = ((x-np.asarray(probe["mean"]))/np.asarray(probe["std"])) @ np.asarray(probe["coef"])+probe["intercept"]
    raw = np.maximum(np.expm1(normalized*scaler["std"][data["atm"]]+scaler["mean"][data["atm"]]), 0.)
    assert np.isfinite(raw).all()
    return raw


def fit_forecast(tx, train, vx, validation, scaler, alphas=ALPHAS):
    mean, std = tx.mean(axis=0), tx.std(axis=0)+1e-8
    standardized = (tx-mean)/std
    states = []
    for alpha in alphas:
        ridge = Ridge(alpha=alpha).fit(standardized, train["target_normalized"])
        probe = dict(mean=mean.tolist(), std=std.tolist(), coef=ridge.coef_.tolist(),
                     intercept=float(ridge.intercept_), alpha=alpha)
        pred = forecast_predict(probe, vx, validation, scaler)
        probe["validation_score"] = float((np.abs(pred-validation["target"])/scaler["mase"][validation["atm"]]).mean())
        states.append(probe)
    best = min(states, key=lambda item: item["validation_score"])
    return dict(best, candidates=[dict(alpha=item["alpha"], validation_score=item["validation_score"]) for item in states])


def bootstrap_indices(series, weeks, repeats=5000, seed=55013, block=4):
    rng = np.random.default_rng(seed)
    atm = rng.integers(0, series, (repeats, series))
    starts = rng.integers(0, weeks, (repeats, (weeks+block-1)//block))
    time = ((starts[:, :, None]+np.arange(block)) % weeks).reshape(repeats, -1)[:, :weeks]
    return atm, time


def bootstrap_ratio(a, b, atm, weeks):
    result = []
    for start in range(0, len(atm), 250):
        rr, tt = atm[start:start+250, :, None], weeks[start:start+250, None, :]
        numerator = a[rr, tt].sum(axis=(1, 2))
        denominator = b[rr, tt].sum(axis=(1, 2))
        if (denominator <= 0).any():
            raise ValueError("Undefined bootstrap ratio; no observations or zero comparator error")
        result.append(numerator/denominator)
    return np.concatenate(result)


def decide(blocks, baseline_name, p=ImputationProtocol()):
    selected = np.mean([blocks[f"neural_{seed}"]["scaled"] for seed in p.initialization_seeds], axis=0).sum(axis=0)
    atm, weeks = bootstrap_indices(selected.shape[0], selected.shape[1], p.bootstrap_repeats, p.bootstrap_seed, p.bootstrap_block)
    results = {}
    for label, method in (("validation_best_simple", baseline_name), ("ridge_gap", "ridge_gap"), ("pca", "pca")):
        reference = blocks[method]["scaled"].sum(axis=0)
        ratio = float(selected.sum()/reference.sum()) if reference.sum() > 0 else None
        try:
            distribution = bootstrap_ratio(selected, reference, atm, weeks)
        except ValueError as exc:
            results[label] = dict(method=method, ratio=ratio,
                                  improvement_pct=100*(1-ratio) if ratio is not None else None,
                                  nominal_two_sided_95pct=None, corrected_one_sided_upper=None,
                                  passed=False, status="undefined", reason=str(exc))
            continue
        upper = float(np.quantile(distribution, 1-.05/3))
        results[label] = dict(method=method, ratio=ratio, improvement_pct=100*(1-ratio),
                              nominal_two_sided_95pct=np.quantile(distribution, [.025, .975]).tolist(),
                              corrected_one_sided_upper=upper, passed=upper < 1.)
    return dict(primary=results, joint_success=all(value["passed"] for value in results.values()),
                scope=f"Additional {100*p.missing_fraction:g}% masked known values; no truth for originally missing cells",
                uncertainty=f"paired series resampling plus shared circular {p.bootstrap_block}-period blocks, conditional on fixed fits/masks"), dict(atm=atm, weeks=weeks)


def plot(root, panel, test, predictions, baseline_name, p=ImputationProtocol()):
    import matplotlib
    matplotlib.use("Agg") # чтобы в ноутбуке графики не ломать
    import matplotlib.pyplot as plt
    import pandas as pd
    from .data import native_values
    from tsprep.provenance import load_npz
    summary = pd.read_csv(root/"imputation_summary.csv")
    names = [name for name in summary.method if not name.startswith("neural_")]+["neural_mean_error"]
    data = summary.set_index("method").loc[names]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, field, label in zip(axes, ("scaled_mae", "native_rmse"), ("Scaled MAE", "RMSE, published units")):
        ax.barh(data.index, data[field], color=["tab:green" if name == "neural_mean_error" else "steelblue" for name in names])
        ax.invert_yaxis()
        ax.set_xlabel(label+" on additionally hidden known values; lower is better")
        ax.grid(axis="x", alpha=.2)
    fig.suptitle(f"Imputation: {100*p.missing_fraction:g}% additional missingness; observed cells are copied")
    fig.savefig(root/"imputation_metrics.png", dpi=150)
    plt.close(fig)
    # Для примера берём первое окно со скрытыми значениями, без выбора по ошибке модели.
    hits = np.flatnonzero(test["score_mask"].any(axis=1))
    row = int(hits[0])
    start = int(test["start"][row])
    dates = (np.asarray(panel["dates"][start:start+p.window], dtype="datetime64[D]")
             if "dates" in panel else np.arange(start, start+p.window))
    scaler = load_npz(root/"scaler.npz")
    known = np.isfinite(test["windows"][row])
    reference = test["truth_raw"][row]
    fig, ax = plt.subplots(figsize=(13, 4), constrained_layout=True)
    ax.plot(dates, reference, color="black", label="recorded reference (hidden values held out)", alpha=.7)
    ax.plot(dates, np.where(known, reference, np.nan), color="tab:blue", label="available input", marker=".")
    for method, color in ((baseline_name, "tab:orange"), (f"neural_{p.initialization_seeds[0]}", "tab:green")):
        reconstructed = native_values(predictions[method], test, scaler)[row]
        ax.plot(dates, reconstructed, label=method, color=color, linestyle="--")
    marked = ~known & np.isfinite(reference)
    ax.scatter(dates[marked], reference[marked], facecolors="none", edgecolors="red", s=70, label="known values additionally hidden")
    ax.set_title("First held-out gap window: imputation")
    ax.set_ylabel("Original units")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=.2)
    fig.savefig(root/"imputation_example.png", dpi=150)
    plt.close(fig)
