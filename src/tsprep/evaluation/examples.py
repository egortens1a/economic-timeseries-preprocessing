"""Frozen-model examples for forecast, classification and anomaly diagnostics."""
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tsprep.experiment import load_artifact
from tsprep.data_preprocessing.datasets import build_eval_loader, forecast_pairs, prepare_and_window
from tsprep.evaluation.reconstruction import collect_embeddings
from tsprep.evaluation.downstream import (forecast_via_linear_probe, anomaly_scores, calibrate_threshold,
                                         window_anomaly_labels)
from tsprep.provenance import read_json


def _ridge(x, y, test, alpha):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(x, y).predict(test)


def downstream_example(directory, *, device="cpu", allow_source_mismatch=False, baselines=None):
    model, p, exp, manifest, arrays = load_artifact(directory, device, allow_source_mismatch=allow_source_mismatch)
    outputs = {split: collect_embeddings(model, build_eval_loader(getattr(p, split+"_windows"), exp.data.batch_size), device)
               for split in ("train", "val", "test")}
    metrics = read_json(Path(directory)/"metrics.json")
    if baselines is None and (Path(directory).parent/"baselines.json").exists():
        baselines = read_json(Path(directory).parent/"baselines.json")
    pairs = {s: forecast_pairs(p, s, exp.data) for s in outputs}
    tr, va, te = [pairs[s] for s in ("train", "val", "test")]
    ztr, zte = outputs["train"]["z"][tr[0]], outputs["test"]["z"][te[0]]
    predictions = dict(embedding_ridge=_ridge(ztr, tr[1], zte, metrics["forecast"]["ridge_alpha"]),
        train_mean=np.full_like(te[1], tr[1].mean()), train_median=np.full_like(te[1], np.median(tr[1])),
        last_observed=outputs["test"]["x"][te[0], -1, exp.data.target_feature])
    flat = {s: out["x"].reshape(len(out["x"]), -1) for s, out in outputs.items()}
    if baselines is not None:
        predictions["raw_window_ridge"] = _ridge(flat["train"][tr[0]], tr[1], flat["test"][te[0]],
            baselines["forecast"]["raw_window_ridge"]["ridge_alpha"])
        pca = PCA(n_components=baselines["pca_components"], svd_solver="full").fit(flat["train"])
        predictions["PCA_ridge"] = _ridge(pca.transform(flat["train"])[tr[0]], tr[1],
            pca.transform(flat["test"])[te[0]], baselines["pca"]["forecast"]["ridge_alpha"])
    reconstructed = {s: out["x_hat"].reshape(len(out["x_hat"]), -1) for s, out in outputs.items()}
    processed_report = forecast_via_linear_probe(reconstructed["train"][tr[0]], tr[1],
        reconstructed["test"][te[0]], te[1], z_val=reconstructed["val"][va[0]], y_val=va[1])
    predictions["reconstructed_window_ridge"] = _ridge(reconstructed["train"][tr[0]], tr[1],
        reconstructed["test"][te[0]], processed_report["ridge_alpha"])
    target_indices = p.window_starts["test"][te[0]]+exp.data.window_len-1+exp.data.forecast_horizon
    endpoint_indices = p.window_starts["test"]+exp.data.window_len-1
    result = dict(model=model, prepared=p, config=exp, manifest=manifest, outputs=outputs,
        forecast=dict(target=te[1], predictions=predictions,
                      time=target_indices if p.dates is None else p.dates[target_indices]),
        endpoint=dict(time=endpoint_indices if p.dates is None else p.dates[endpoint_indices],
                      observed=outputs["test"]["x"][:, -1], reconstructed=outputs["test"]["x_hat"][:, -1]),
        reconstructed_forecast=processed_report, classification=None, anomaly=None)
    if baselines is not None:
        result["endpoint"]["PCA"] = pca.inverse_transform(pca.transform(flat["test"])).reshape(p.test_windows.shape)[:, -1]
    if p.regime is not None:
        labels = {s: p.regime[p.window_starts[s]+exp.data.window_len-1] for s in ("train", "test")}
        valid = {s: np.isfinite(values) for s, values in labels.items()}
        ytr, yte = [labels[s][valid[s]] for s in ("train", "test")]
        if np.isin(ytr, [0, 1]).all() and len(np.unique(ytr)) == 2 and len(yte):
            classifier = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
            classifier.fit(outputs["train"]["z"][valid["train"]], ytr)
            scores = classifier.predict_proba(outputs["test"]["z"][valid["test"]])[:, 1]
            result["classification"] = dict(labels=yte, scores=scores, prediction=(scores >= .5).astype(int))
    if p.is_anomaly is not None:
        val, test = outputs["val"], outputs["test"]
        lam = metrics.get("anomaly", {}).get("kl_lambda", .1)
        quantile = metrics.get("anomaly", {}).get("quantile", .95)
        val_scores = anomaly_scores(val["x"], val["x_hat"], val["kl"], lam)
        scores = anomaly_scores(test["x"], test["x_hat"], test["kl"], lam)
        val_point = ((val["x"][:, -1]-val["x_hat"][:, -1])**2).mean(axis=1)
        point = ((test["x"][:, -1]-test["x_hat"][:, -1])**2).mean(axis=1)
        lo, hi = p.split_bounds["test"]
        result["anomaly"] = dict(window_scores=scores, window_threshold=calibrate_threshold(val_scores, quantile=quantile),
            validation_window_scores=val_scores, validation_endpoint_scores=val_point,
            window_labels=window_anomaly_labels(p.is_anomaly[lo:hi], exp.data.window_len, exp.data.stride, len(scores)),
            endpoint_scores=point, endpoint_threshold=calibrate_threshold(val_point, quantile=quantile),
            endpoint_labels=p.is_anomaly[endpoint_indices].astype(int), quantile=quantile)
    if "clean_raw" in arrays:
        clean = prepare_and_window(arrays["clean_raw"], exp.data, normalizer=p.normalizer)
        result["endpoint"]["clean"] = clean.test_windows[:, -1]
    return result
