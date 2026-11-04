"""Tabular views of saved reconstruction and downstream measurements."""
from pathlib import Path
import numpy as np
import pandas as pd
from tsprep.config import DataConfig
from tsprep.data_preprocessing.datasets import prepare_and_window
from tsprep.data_preprocessing.preprocessing import TrainFittedNormalizer
from tsprep.evaluation.downstream import (binary_report, event_report, window_anomaly_labels,
                                         anomaly_scores, calibrate_threshold)
from tsprep.provenance import read_json, load_npz
from tsprep.study import validate_completed_study


def prepared_snapshot(artifact):
    artifact = Path(artifact)
    manifest = read_json(artifact/"manifest.json")
    arrays = load_npz(artifact/"input.npz")
    cfg = DataConfig(**manifest["config"]["data"])
    p = prepare_and_window(arrays["raw"], cfg, arrays.get("regime"), arrays.get("is_anomaly"),
        dates=arrays.get("dates"), feature_names=manifest["preprocessing"]["features"],
        normalizer=TrainFittedNormalizer.from_state(manifest["preprocessing"]["normalizer"]))
    return p, cfg, arrays, manifest


def _record(identity, result, **extra):
    return {**identity, **{key: value for key, value in result.items() if not isinstance(value, (dict, list))}, **extra}


def _support(p, cfg, dataset, fold):
    rows = []
    for split in ("train", "val", "test"):
        starts = p.window_starts[split]
        ends = starts+cfg.window_len-1
        for task, labels in (("regime_endpoint", None if p.regime is None else p.regime[ends]),
                             ("anomaly_endpoint", None if p.is_anomaly is None else p.is_anomaly[ends])):
            if labels is None:
                continue
            valid = np.isfinite(labels)
            for label, count in zip(*np.unique(labels[valid], return_counts=True)):
                rows.append(dict(dataset=dataset, fold=fold, split=split, task=task,
                    label=int(label), count=int(count), n_observed=len(labels), n_missing=int((~valid).sum())))
        if p.is_anomaly is not None:
            lo, hi = p.split_bounds[split]
            labels = window_anomaly_labels(p.is_anomaly[lo:hi], cfg.window_len, cfg.stride, len(starts))
            for label in (0, 1):
                rows.append(dict(dataset=dataset, fold=fold, split=split, task="anomaly_window", label=label,
                    count=int((labels == label).sum()), n_observed=len(labels), n_missing=0))
    return rows


def _raw_anomaly(p, cfg):
    reports = {}
    for unit in ("window", "endpoint"):
        va = p.val_windows if unit == "window" else p.val_windows[:, -1:]
        te = p.test_windows if unit == "window" else p.test_windows[:, -1:]
        threshold = calibrate_threshold(anomaly_scores(va, np.zeros_like(va)))
        scores = anomaly_scores(te, np.zeros_like(te))
        if unit == "window":
            lo, hi = p.split_bounds["test"]
            labels = window_anomaly_labels(p.is_anomaly[lo:hi], cfg.window_len, cfg.stride, len(te))
        else:
            labels = p.is_anomaly[p.window_starts["test"]+cfg.window_len-1].astype(int)
        result = dict(**binary_report(labels, (scores > threshold).astype(int), scores), threshold=threshold,
            calibration="unlabelled_validation_quantile", quantile=.95, kl_lambda=0.,
            predicted_positive_rate=float((scores > threshold).mean()))
        if unit == "endpoint":
            result["events_on_observed_endpoints"] = event_report(labels, scores > threshold)
        reports[unit] = result
    return reports


def load_study_tables(directory, *, allow_source_mismatch=False):
    """Read saved metrics without retraining models or overwriting artifacts."""
    directory = Path(directory)
    runs = validate_completed_study(directory, allow_legacy=allow_source_mismatch,
                                    allow_source_mismatch=allow_source_mismatch)
    records = {key: [] for key in ("forecast", "classification", "clustering", "anomaly", "clean",
                                  "reconstruction", "latent", "support", "availability")}
    groups = {}

    def add(identity, metrics):
        records["forecast"].append(_record(identity, metrics["forecast"]))
        records["reconstruction"].append(_record(identity, {k: metrics[k] for k in ("mse", "mae", "r2")}))
        for task in ("classification", "clustering"):
            if task in metrics:
                records[task].append(_record(identity, metrics[task]))
        for key, unit in (("anomaly", "window"), ("anomaly_endpoint", "endpoint")):
            if key in metrics:
                result = metrics[key]
                events = result.get("events_on_observed_endpoints", {})
                records["anomaly"].append(_record(identity, result, unit=unit, **events))
        if "clean_reconstruction" in metrics:
            records["clean"].append(_record(identity, metrics["clean_reconstruction"], weighting="all_window_elements"))
        if "latent" in metrics:
            records["latent"].append(_record(identity, metrics["latent"], kl=metrics.get("test_kl_mean"),
                variance_ratio=metrics.get("variance_ratio_scale_dependent")))

    for row in runs.itertuples():
        identity = dict(dataset=row.dataset, fold=row.fold, model=row.model, seed=row.seed, role="model")
        metrics = read_json(directory/row.artifact/"metrics.json")
        add(identity, metrics)
        key = (row.dataset, row.fold)
        if key in groups:
            continue
        base = read_json(directory/row.dataset/f"fold_{row.fold}"/"baselines.json")
        groups[key] = base
        p, cfg, arrays, _ = prepared_snapshot(directory/row.artifact)
        records["support"].extend(_support(p, cfg, *key))
        finite = np.array([]) if p.regime is None else p.regime[np.isfinite(p.regime)]
        binary = len(finite) > 0 and np.isin(finite, [0, 1]).all()
        records["availability"].append(dict(dataset=row.dataset, fold=row.fold,
            forecast=True, classification=binary, true_regime=p.regime is not None,
            anomaly=p.is_anomaly is not None, clean="clean_raw" in arrays,
            horizon=cfg.forecast_horizon, target_feature=p.feature_names[cfg.target_feature],
            target_kind="single_future_transformed_observation", stride=cfg.stride))
        baseline_id = {**identity, "seed": None, "role": "baseline"}
        add({**baseline_id, "model": "PCA"}, base["pca"])
        records["reconstruction"].append(_record({**baseline_id, "model": "zero"}, base["zero_reconstruction"]))
        for name, report in base["forecast"].items():
            records["forecast"].append(_record({**baseline_id, "model": name}, report))
        if "noisy_input_vs_clean" in metrics:
            records["clean"].append(_record({**baseline_id, "model": "observed_input"},
                                            metrics["noisy_input_vs_clean"], weighting="all_window_elements"))
        if binary:
            labels = {s: p.regime[p.window_starts[s]+cfg.window_len-1] for s in ("train", "test")}
            tr = labels["train"][np.isfinite(labels["train"])].astype(int)
            te = labels["test"][np.isfinite(labels["test"])].astype(int)
            if len(tr) and len(te):
                majority = int(np.bincount(tr, minlength=2).argmax())
                result = binary_report(te, np.full_like(te, majority), np.full(te.shape, tr.mean()))
                records["classification"].append(_record({**baseline_id, "model": "train_majority"}, result,
                                                           positive_rate_train=float(tr.mean())))
        if p.is_anomaly is not None:
            for unit, report in _raw_anomaly(p, cfg).items():
                events = report.get("events_on_observed_endpoints", {})
                records["anomaly"].append(_record({**baseline_id, "model": "raw_magnitude"}, report,
                                                   unit=unit, **events))
    tables = {key: pd.DataFrame(value) for key, value in records.items()}
    tables["runs"] = runs
    forecast = tables["forecast"]
    for name, key in (("mean", "train_mean"), ("median", "train_median"),
                      ("raw", "raw_window_ridge"), ("pca", "PCA")):
        reference = forecast[(forecast.role == "baseline") & (forecast.model == key)].set_index(["dataset", "fold"]).mae
        forecast["delta_mae_vs_"+name] = [value-reference.loc[(dataset, fold)]
            for dataset, fold, value in forecast[["dataset", "fold", "mae"]].itertuples(index=False, name=None)]
    clean = tables["clean"]
    if not clean.empty:
        for metric in ("mse", "mae"):
            reference = clean[clean.model == "observed_input"].set_index(["dataset", "fold"])[metric]
            clean["delta_"+metric+"_vs_input"] = [value-reference.loc[(dataset, fold)]
                for dataset, fold, value in clean[["dataset", "fold", metric]].itertuples(index=False, name=None)]
    return tables


def task_summary(frame, metrics):
    """Summarize initialization seeds within each fold; keep absent tasks empty."""
    columns = [column for column in metrics if column in frame]
    if frame.empty or not columns:
        return pd.DataFrame()
    groups = [column for column in ("dataset", "fold", "model", "role", "unit", "label_source") if column in frame]
    return frame.groupby(groups, dropna=False)[columns].agg(["mean", "std", "count"])
