"""One train/save/load/evaluate protocol shared by CLI and notebooks."""
from dataclasses import asdict
from pathlib import Path
import json
import time
import warnings
import numpy as np
import sklearn
import torch
from sklearn.decomposition import PCA
from tsprep.config import DataConfig, ModelConfig, TrainConfig, ExperimentConfig
from tsprep.data_preprocessing.datasets import (prepare_and_window, build_dataloaders,
    build_eval_loader, forecast_pairs, aggregate_windows)
from tsprep.data_preprocessing.preprocessing import TrainFittedNormalizer, prepare_series
from tsprep.evaluation.reconstruction import collect_embeddings, reconstruction_report
from tsprep.evaluation.downstream import (forecast_via_linear_probe, regression_report, classify_via_linear_probe,
    clustering_quality, anomaly_scores, calibrate_threshold, anomaly_detection_f1, window_anomaly_labels,
    binary_report, event_report)
from tsprep.evaluation.latent_space import latent_diagnostics, information_collapse_ratio
from tsprep.models.autoencoder import build_model
from tsprep.mlcore.trainer import train_model, seed_everything
from tsprep.provenance import (write_json, read_json, file_hash, source_hashes, computational_hashes,
    value_hash, evaluator_hash, input_arrays, input_fingerprint, save_npz, load_npz,
    source_snapshot, runtime_environment)

ARTIFACT_VERSION = 3


def metric_binding(directory):
    return dict(manifest_sha256=file_hash(Path(directory)/"manifest.json"), evaluator_sha256=evaluator_hash())


def save_metrics(directory, result):
    target = Path(directory)
    result = {**result, "provenance": metric_binding(target)}
    write_json(target/"metrics.json", result)
    write_json(target/"metrics_receipt.json", dict(sha256=file_hash(target/"metrics.json"),
                                                  **result["provenance"]))
    return result


def verify_metrics(directory, *, allow_evaluator_mismatch=False):
    target = Path(directory)
    receipt = read_json(target/"metrics_receipt.json")
    binding = {key: receipt.get(key) for key in ("manifest_sha256", "evaluator_sha256")}
    if receipt.get("sha256") != file_hash(target/"metrics.json") or binding["manifest_sha256"] != file_hash(target/"manifest.json"):
        raise ValueError(f"Cached metrics provenance/checksum mismatch: {target}")
    if binding["evaluator_sha256"] != evaluator_hash() and not allow_evaluator_mismatch:
        raise ValueError(f"Cached metrics evaluator mismatch: {target}")
    result = read_json(target/"metrics.json")
    if result.get("provenance") != binding:
        raise ValueError(f"Cached metrics binding mismatch: {target}")
    return result


def verify_artifact(directory, *, expected_config=None, expected_fingerprint=None,
                    allow_source_mismatch=False):
    target = Path(directory)
    manifest = read_json(target/"manifest.json")
    if manifest["version"] not in (2, ARTIFACT_VERSION):
        raise ValueError("Unsupported artifact version")
    for name in ("input.npz", "model.pt"):
        if name not in manifest["hashes"]:
            raise ValueError(f"Missing required artifact checksum: {name}")
    for name, expected in manifest["hashes"].items():
        if file_hash(target/name) != expected:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    mismatch = computational_hashes(manifest["source_hashes"]) != computational_hashes()
    if mismatch and not allow_source_mismatch:
        raise ValueError("Computational source mismatch; use the archived source or explicitly allow historical reevaluation")
    if mismatch:
        warnings.warn("Historical artifact evaluated with different source; original interpolation is preserved", UserWarning)
    if expected_config is not None and value_hash(manifest["config"]) != value_hash(expected_config):
        raise ValueError("Cached artifact configuration mismatch")
    arrays = load_npz(target/"input.npz")
    fingerprint = input_fingerprint(arrays, manifest["preprocessing"]["features"])
    if manifest.get("input_fingerprint", fingerprint) != fingerprint:
        raise ValueError("Artifact input fingerprint mismatch")
    if expected_fingerprint is not None and expected_fingerprint != fingerprint:
        raise ValueError("Cached artifact belongs to different inputs")
    return manifest


def preprocessing_metadata(p):
    return dict(normalizer=p.normalizer.state(), y0=p.y0.tolist(), seasonal_period=p.seasonal_period,
                log_transform=p.log_transform, features=p.feature_names, split_bounds=p.split_bounds,
                window_counts={s: len(getattr(p, s+"_windows")) for s in p.split_bounds},
                split_dates={s: [str(p.dates[a]), str(p.dates[b-1])] for s, (a,b) in p.split_bounds.items()}
                    if p.dates is not None else None)


def _out_from_arrays(windows, z, reconstructed):
    return dict(x=np.asarray(windows, dtype=np.float32), z=z, x_hat=reconstructed,
                kl=np.zeros(len(windows)))


def evaluate_outputs(outputs, p, cfg, *, anomaly_quantile=.95, anomaly_lambda=.1):
    test = outputs["test"]
    result = {**reconstruction_report(test["x"], test["x_hat"]),
        "variance_ratio_scale_dependent": information_collapse_ratio(test["z"], test["x"]),
        "latent": latent_diagnostics(test["z"]), "test_kl_mean": float(test["kl"].mean()),
        "reconstruction_unit": "standardized_transformed_feature",
        "metric_weighting": "all window elements; overlaps are repeated",
        "target": {"feature": p.feature_names[cfg.target_feature], "horizon_observations": cfg.forecast_horizon,
                   "kind": "single_future_transformed_observation", "log_transform": cfg.log_transform,
                   "difference_period": cfg.seasonal_period}}
    pairs = {s: forecast_pairs(p, s, cfg) for s in outputs}
    tr, va, te = [pairs[s] for s in ("train", "val", "test")]
    result["forecast"] = forecast_via_linear_probe(outputs["train"]["z"][tr[0]], tr[1],
        test["z"][te[0]], te[1], z_val=outputs["val"]["z"][va[0]], y_val=va[1])
    if p.regime is not None:
        labels = {s: p.regime[p.window_starts[s]+cfg.window_len-1] for s in outputs}
        result["clustering"] = clustering_quality(test["z"], labels["test"])
        finite = np.r_[p.regime[np.isfinite(p.regime)]]
        if np.isin(finite, [0, 1]).all():
            result["classification"] = classify_via_linear_probe(outputs["train"]["z"], labels["train"],
                                                                test["z"], labels["test"])
    else:
        result["clustering"] = clustering_quality(test["z"])
    if p.is_anomaly is not None:
        labels = {}
        for s in ("val", "test"):
            a,b = p.split_bounds[s]
            labels[s] = window_anomaly_labels(p.is_anomaly[a:b], cfg.window_len, cfg.stride, len(outputs[s]["x"]))
        val = outputs["val"]
        scores = anomaly_scores(val["x"], val["x_hat"], val["kl"], anomaly_lambda)
        threshold = calibrate_threshold(scores, quantile=anomaly_quantile)
        result["anomaly"] = anomaly_detection_f1(test["x"], test["x_hat"], labels["test"],
            threshold=threshold, kl_per_window=test["kl"], lam=anomaly_lambda)
        result["anomaly"].update(calibration="unlabelled_validation_quantile",
                                 quantile=anomaly_quantile, kl_lambda=anomaly_lambda, label_unit="contaminated_window")
        # Endpoint scores use only each observed window, no retrospective overlap averaging.
        val_point = np.mean((val["x"][:, -1]-val["x_hat"][:, -1])**2, axis=1)
        test_point = np.mean((test["x"][:, -1]-test["x_hat"][:, -1])**2, axis=1)
        threshold_point = calibrate_threshold(val_point, quantile=anomaly_quantile)
        point_labels = p.is_anomaly[p.window_starts["test"]+cfg.window_len-1].astype(int)
        result["anomaly_endpoint"] = {**binary_report(point_labels, (test_point>threshold_point).astype(int), test_point),
            "threshold": threshold_point, "stride": cfg.stride,
            "events_on_observed_endpoints": event_report(point_labels, test_point>threshold_point)}
    return result


def evaluate_model(model, prepared, cfg, device="cpu", clean_raw=None):
    start = time.perf_counter()
    outputs = {s: collect_embeddings(model, build_eval_loader(getattr(prepared, s+"_windows"), cfg.batch_size),
                                    device=device) for s in ("train", "val", "test")}
    result = evaluate_outputs(outputs, prepared, cfg)
    result["evaluation_seconds"] = time.perf_counter()-start
    if clean_raw is not None:
        clean = prepare_and_window(clean_raw, cfg, normalizer=prepared.normalizer)
        result["clean_reconstruction"] = reconstruction_report(clean.test_windows, outputs["test"]["x_hat"])
        result["noisy_input_vs_clean"] = reconstruction_report(clean.test_windows, outputs["test"]["x"])
    return result


def evaluate_baselines(p, cfg, latent_dim=16, clean_raw=None):
    windows = {s: np.asarray(getattr(p, s+"_windows"), dtype=np.float32) for s in ("train", "val", "test")}
    flat = {s: w.reshape(len(w), -1) for s,w in windows.items()}
    n_components = min(latent_dim, *flat["train"].shape)
    pca = PCA(n_components=n_components, svd_solver="full").fit(flat["train"])
    z = {s: pca.transform(f) for s,f in flat.items()}
    outputs = {s: _out_from_arrays(windows[s], z[s], pca.inverse_transform(z[s]).reshape(windows[s].shape))
               for s in windows}
    result = {"pca": evaluate_outputs(outputs, p, cfg), "pca_components": n_components,
              "zero_reconstruction": reconstruction_report(windows["test"], np.zeros_like(windows["test"]))}
    pairs = {s: forecast_pairs(p, s, cfg) for s in windows}
    tr,va,te = [pairs[s] for s in ("train", "val", "test")]
    y = te[1]
    result["forecast"] = {
        "train_mean": regression_report(y, np.full_like(y, tr[1].mean())),
        "train_median": regression_report(y, np.full_like(y, np.median(tr[1]))),
        "last_observed_transformed_value": regression_report(y, windows["test"][te[0], -1, cfg.target_feature]),
        "raw_window_ridge": forecast_via_linear_probe(flat["train"][tr[0]], tr[1], flat["test"][te[0]], y,
                           z_val=flat["val"][va[0]], y_val=va[1])}
    if clean_raw is not None:
        clean = prepare_and_window(clean_raw, cfg, normalizer=p.normalizer)
        result["pca"]["clean_reconstruction"] = reconstruction_report(clean.test_windows, outputs["test"]["x_hat"])
    return result


def train_experiment(exp, raw, *, output_dir, regime=None, is_anomaly=None, dates=None, feature_names=None,
                     clean_raw=None, metadata=None, verbose=True, evaluate=True):
    """Persist immutable inputs, full settings and fitted preprocessing before evaluation."""
    needs_clean = "clean" in (exp.train.training_target, exp.train.validation_target)
    if needs_clean and (clean_raw is None or np.shape(clean_raw) != np.shape(raw)):
        raise ValueError("Clean training/validation target requires aligned clean_raw")
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=False)
    seed_everything(exp.train.seed, exp.train.deterministic)
    prepared = prepare_and_window(raw, exp.data, regime, is_anomaly, dates=dates, feature_names=feature_names)
    model = build_model(exp.model, exp.data.n_features, exp.data.window_len)
    clean = prepare_and_window(clean_raw, exp.data, normalizer=prepared.normalizer) if needs_clean else None
    train, val, _ = build_dataloaders(prepared, exp.data,
        train_targets=clean.train_windows if exp.train.training_target == "clean" else None,
        val_targets=clean.val_windows if exp.train.validation_target == "clean" else None)
    history = train_model(model, train, val, exp.train, verbose=verbose)
    model.to("cpu")
    torch.save(model.state_dict(), target/"model.pt")
    arrays = input_arrays(raw, regime=regime, is_anomaly=is_anomaly, clean_raw=clean_raw, dates=dates)
    save_npz(target/"input.npz", arrays)
    source_snapshot(target/"source.zip")
    manifest = dict(version=ARTIFACT_VERSION, config=asdict(exp), preprocessing=preprocessing_metadata(prepared),
        metadata=metadata or {}, history=asdict(history),
        parameters=sum(p.numel() for p in model.parameters()), source_hashes=source_hashes(),
        input_fingerprint=input_fingerprint(arrays, prepared.feature_names),
        environment=runtime_environment(exp.train.device),
        hashes={name: file_hash(target/name) for name in ("input.npz", "model.pt", "source.zip")})
    write_json(target/"manifest.json", manifest)
    if evaluate:
        result = evaluate_model(model, prepared, exp.data, exp.train.device, clean_raw)
        result.update(model=exp.model.model_name, seed=exp.train.seed, best_epoch=history.best_epoch,
                      best_beta=history.best_beta, validation_score=history.best_score,
                      epochs_ran=len(history.train_loss), parameters=manifest["parameters"],
                      training_seconds=float(sum(history.epoch_time_sec)))
        return save_metrics(target, result)
    return {"validation_score": history.best_score, "artifact": str(target)}


def load_artifact(directory, device="cpu", *, allow_source_mismatch=False):
    target = Path(directory)
    manifest = verify_artifact(target, allow_source_mismatch=allow_source_mismatch)
    config = manifest["config"]
    model_config = dict(config["model"])
    if manifest["version"] == 2:
        model_config.setdefault("upsampling", "linear")
    exp = ExperimentConfig(DataConfig(**config["data"]), ModelConfig(**model_config), TrainConfig(**config["train"]))
    arrays = load_npz(target/"input.npz")
    norm = TrainFittedNormalizer.from_state(manifest["preprocessing"]["normalizer"])
    prepared = prepare_and_window(arrays["raw"], exp.data, arrays.get("regime"), arrays.get("is_anomaly"),
        dates=arrays.get("dates"), feature_names=manifest["preprocessing"]["features"], normalizer=norm)
    if preprocessing_metadata(prepared) != manifest["preprocessing"]:
        # JSON stores tuple bounds as lists.
        if json.loads(json.dumps(preprocessing_metadata(prepared))) != manifest["preprocessing"]:
            raise ValueError("Preprocessing metadata mismatch")
    model = build_model(exp.model, exp.data.n_features, exp.data.window_len)
    model.load_state_dict(torch.load(target/"model.pt", map_location="cpu", weights_only=True))
    model.to(device).eval()
    return model, prepared, exp, manifest, arrays


def evaluate_artifact(directory, device="cpu", *, allow_source_mismatch=False):
    model, p, exp, manifest, arrays = load_artifact(directory, device, allow_source_mismatch=allow_source_mismatch)
    result = evaluate_model(model, p, exp.data, device, arrays.get("clean_raw"))
    result.update(model=exp.model.model_name, seed=exp.train.seed, best_epoch=manifest["history"]["best_epoch"],
        best_beta=manifest["history"]["best_beta"], validation_score=manifest["history"]["best_score"],
        epochs_ran=len(manifest["history"]["train_loss"]), parameters=manifest["parameters"],
        training_seconds=float(sum(manifest["history"]["epoch_time_sec"])))
    return result
