"""Immutable study inputs, verified resume and transactional final tables."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pandas as pd
import torch
from tsprep.config import (ALL_MODEL_NAMES, ABLATION_MODEL_NAMES, DataConfig, ModelConfig,
                           TrainConfig, ExperimentConfig, positive_int)
from tsprep.data_preprocessing.synthetic import generate_series
from tsprep.data_preprocessing.loaders import load_series_with_recession_labels, load_csv_series_with_dates
from tsprep.data_preprocessing.datasets import prepare_and_window
from tsprep.data_preprocessing.quality import csv_quality_report
from tsprep.experiment import (train_experiment, evaluate_artifact, evaluate_baselines,
    preprocessing_metadata, verify_artifact, verify_metrics, save_metrics)
from tsprep.mlcore.trainer import seed_everything
from tsprep.provenance import (write_json, read_json, file_hash, computational_hashes, input_arrays,
    input_fingerprint, value_hash, save_npz, load_npz, atomic_write, runtime_environment,
    source_snapshot, study_lock, evaluator_hash)

STUDY_VERSION = 3
GRID_COLUMNS = ["dataset", "fold", "model", "seed"]


def dataset(root, name, *, data_seed=2026):
    if name == "synthetic":
        s = generate_series(length=3000, seed=data_seed)
        return dict(raw=s.values, regime=s.regime, is_anomaly=s.is_anomaly, clean_raw=s.clean_values,
                    dates=None, feature_names=["synthetic_price"])
    if name == "aapl":
        b = load_series_with_recession_labels(str(root/"data/raw/aapl.csv"), str(root/"data/raw/USREC.csv"),
                                             value_cols=["Adj Close"])
        return dict(raw=b["values"], regime=b["labels"], dates=b["dates"], feature_names=b["value_cols"])
    if name == "btc":
        dates, raw, cols = load_csv_series_with_dates(str(root/"data/raw/btc_usd.csv"), value_cols=["Close"])
        return dict(raw=raw, dates=dates, feature_names=cols)
    raise ValueError(name)


def fold_data(data, fold):
    if fold not in (0, 1):
        raise ValueError("fold must be 0 or 1")
    end = int(len(data["raw"])*.85) if fold == 0 else len(data["raw"])
    return {k: v[:end] if isinstance(v, np.ndarray) else v for k, v in data.items()}


def data_arrays(data):
    return input_arrays(**{k: v for k, v in data.items() if k != "feature_names"})


def expected_grid(protocol):
    return set(product(protocol["datasets"], protocol["folds"], protocol["models"], protocol["seeds"]))


def check_grid(frame, protocol):
    if not set(GRID_COLUMNS).issubset(frame.columns) or frame[GRID_COLUMNS].isna().any().any():
        raise ValueError("Missing study grid columns/values")
    actual = list(frame[GRID_COLUMNS].itertuples(index=False, name=None))
    if len(actual) != len(set(actual)) or set(actual) != expected_grid(protocol):
        raise ValueError("Incomplete or duplicated dataset/fold/model/seed grid")


def summarize(rows, target):
    frame = pd.DataFrame(rows)
    atomic_write(target/"comparison_runs.csv", frame.to_csv(index=False).encode("utf-8"))
    columns = [c for c in ("mse", "forecast_mae", "forecast_vs_mean", "forecast_vs_median", "forecast_vs_raw",
        "forecast_vs_pca", "mse_vs_pca", "clean_mse", "clean_mae", "input_clean_mse", "input_clean_mae",
        "training_seconds") if c in frame]
    summary = frame.groupby(["dataset", "model"])[columns].agg(["mean", "std", "min", "max"])
    atomic_write(target/"comparison_summary.csv", summary.to_csv().encode("utf-8"))


def validate_completed_study(target, *, allow_legacy=False):
    """Validate grid, all candidates and selected metrics, including old studies explicitly."""
    target = Path(target)
    protocol, complete = read_json(target/"protocol.json"), read_json(target/"complete.json")
    legacy = protocol.get("version", 2) != STUDY_VERSION
    if legacy and not allow_legacy:
        raise ValueError("Historical study requires explicit allow_legacy; resume is not supported")
    frame = pd.read_csv(target/"comparison_runs.csv")
    check_grid(frame, protocol)
    if complete["runs"] != len(frame):
        raise ValueError("Completion row count mismatch")
    if not legacy:
        if not {"protocol.json", "comparison_runs.csv", "comparison_summary.csv"} <= set(complete.get("files", {})):
            raise ValueError("Incomplete completion receipt")
        for name, expected in complete["files"].items():
            if file_hash(target/name) != expected:
                raise ValueError(f"Completion checksum mismatch: {name}")
    identities = {}
    for path in sorted(target.glob("*/fold_*/*/manifest.json")):
        m = verify_artifact(path.parent, allow_source_mismatch=allow_legacy)
        fingerprint = input_fingerprint(load_npz(path.parent/"input.npz"), m["preprocessing"]["features"])
        identities.setdefault(path.parent.parent, set()).add(fingerprint)
    if any(len(versions) != 1 for versions in identities.values()):
        raise ValueError("Mixed input snapshots in a study fold")
    for row in frame.itertuples():
        artifact = (target/row.artifact).resolve()
        if not artifact.is_relative_to(target.resolve()):
            raise ValueError("Artifact path escapes study")
        m = read_json(artifact/"manifest.json")
        if m["config"]["model"]["model_name"] != row.model or m["config"]["train"]["seed"] != row.seed:
            raise ValueError("Table artifact configuration mismatch")
        metrics = read_json(artifact/"metrics.json") if legacy else verify_metrics(artifact)
        if not np.isclose(metrics["mse"], row.mse, rtol=0, atol=1e-12) or not np.isclose(
                metrics["forecast"]["mae"], row.forecast_mae, rtol=0, atol=1e-12):
            raise ValueError("Table metrics mismatch")
    return frame


def _baseline(group, data, cfg, fingerprint):
    receipt_path = group/"baseline_receipt.json"
    binding = dict(input_fingerprint=fingerprint, config=asdict(cfg), latent_dim=16,
                   evaluator_sha256=evaluator_hash())
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if receipt["binding"] != binding:
            raise ValueError("Baseline data/preprocessing/evaluator mismatch")
        for name, expected in receipt["hashes"].items():
            if file_hash(group/name) != expected:
                raise ValueError(f"Baseline checksum mismatch: {name}")
        return read_json(group/"baselines.json")
    if (group/"baselines.json").exists():
        raise RuntimeError(f"Incomplete baseline receipt: {group}")
    prepared = prepare_and_window(data["raw"], cfg, data.get("regime"), data.get("is_anomaly"),
                                 dates=data.get("dates"), feature_names=data["feature_names"])
    result = evaluate_baselines(prepared, cfg, clean_raw=data.get("clean_raw"))
    write_json(group/"preprocessing.json", preprocessing_metadata(prepared))
    write_json(group/"baselines.json", result)
    write_json(receipt_path, dict(binding=binding, hashes={name: file_hash(group/name)
                                 for name in ("baselines.json", "preprocessing.json")}))
    return result


def run_study(root, target, *, epochs=30, seeds=(42, 43, 44), datasets=("synthetic", "aapl", "btc"),
              models=None, folds=(0, 1), device="cpu", beta_grid=(.001, .01), data_seed=2026,
              deterministic=True):
    root, target = Path(root).resolve(), Path(target).resolve()
    models = list(ALL_MODEL_NAMES + ABLATION_MODEL_NAMES if models is None else models)
    positive_int("epochs", epochs)
    for name, values in (("seeds", seeds), ("datasets", datasets), ("models", models),
                         ("folds", folds), ("beta_grid", beta_grid)):
        if not values or len(set(values)) != len(values):
            raise ValueError(f"{name} must be nonempty and unique")
    if not set(models) <= set(ALL_MODEL_NAMES + ABLATION_MODEL_NAMES) or not set(folds) <= {0, 1}:
        raise ValueError("Invalid model or fold")
    if any(not np.isfinite(beta) or beta < 0 for beta in beta_grid):
        raise ValueError("Invalid beta grid")
    seed_everything(seeds[0], deterministic)
    quality = {p.name: csv_quality_report(p) for p in sorted((root/"data/raw").glob("*.csv"))}
    originals = {}
    for name in datasets:
        if name != "synthetic" and quality[{"aapl": "aapl.csv", "btc": "btc_usd.csv"}[name]]["needs_review"]:
            raise ValueError(f"Quality gate failed for {name}")
        originals[name] = dataset(root, name, data_seed=data_seed)
    fingerprints = {name: input_fingerprint(data_arrays(data), data["feature_names"])
                    for name, data in originals.items()}
    protocol = dict(version=STUDY_VERSION, epochs=epochs, seeds=list(seeds), datasets=list(datasets),
        models=models, folds=list(folds), device=device, beta_grid=list(beta_grid), data_seed=data_seed,
        deterministic=deterministic, input_fingerprints=fingerprints,
        beta_selection="validation reconstruction, first seed per fold", window_len=48, stride=2,
        hidden_dim=32, latent_dim=16, tcn_levels=3, upsampling="matmul",
        checkpoint="minimum deterministic validation reconstruction after KL warm-up",
        anomaly_quantile=.95, source_hashes=computational_hashes(), environment=runtime_environment(device))
    if (target/"protocol.json").exists():
        if read_json(target/"protocol.json") != protocol:
            raise ValueError("Cannot resume with changed inputs, protocol, environment or computational source")
    else:
        target.mkdir(parents=True, exist_ok=False)
    with study_lock(target):
        if not (target/"protocol.json").exists():
            write_json(target/"protocol.json", protocol)
            write_json(target/"data_quality.json", quality)
            source_snapshot(target/"source.zip")
            freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
            atomic_write(target/"environment.txt", freeze.stdout.encode("utf-8"))
            for name, data in originals.items():
                save_npz(target/"inputs"/f"{name}.npz", data_arrays(data))
                write_json(target/"inputs"/f"{name}.json", dict(features=data["feature_names"],
                    fingerprint=fingerprints[name], sha256=file_hash(target/"inputs"/f"{name}.npz")))
        snapshots = {}
        for name in datasets:
            receipt = read_json(target/"inputs"/f"{name}.json")
            arrays = load_npz(target/"inputs"/f"{name}.npz")
            if (receipt["sha256"] != file_hash(target/"inputs"/f"{name}.npz") or
                    input_fingerprint(arrays, receipt["features"]) != fingerprints[name]):
                raise ValueError("Study input snapshot checksum/fingerprint mismatch")
            snapshots[name] = dict(**arrays, feature_names=receipt["features"])
        if (target/"complete.json").exists():
            validate_completed_study(target)
            return target
        started = time.perf_counter()
        write_json(target/"state.json", dict(status="running", protocol_sha256=value_hash(protocol)))
        try:
            rows = _execute(snapshots, target, protocol, started)
            check_grid(pd.DataFrame(rows), protocol)
            summarize(rows, target)
            files = [target/"protocol.json", target/"data_quality.json", target/"environment.txt",
                     target/"source.zip", target/"comparison_runs.csv", target/"comparison_summary.csv"]
            files += list((target/"inputs").glob("*"))
            files += list(target.glob("*/fold_*/*.json")) + list(target.glob("*/fold_*/*/*.json"))
            write_json(target/"complete.json", dict(version=1, runs=len(rows),
                wall_seconds=time.perf_counter()-started,
                files={path.relative_to(target).as_posix(): file_hash(path) for path in sorted(files)}))
            write_json(target/"state.json", dict(status="complete", runs=len(rows)))
        except BaseException as exc:
            write_json(target/"state.json", dict(status="interrupted", error=type(exc).__name__, message=str(exc)))
            raise
    return target


def _row(name, fold, model, seed, beta, score, selected, result, baseline, target):
    row = dict(dataset=name, fold=fold, model=model, seed=seed, beta=beta, validation_score=score,
        mse=result["mse"], r2=result["r2"], forecast_mae=result["forecast"]["mae"],
        forecast_r2=result["forecast"]["r2"], mse_vs_pca=result["mse"]-baseline["pca"]["mse"],
        epochs=result["epochs_ran"], best_epoch=result["best_epoch"], training_seconds=result["training_seconds"],
        parameters=result["parameters"], active_units=result["latent"]["active_units"], kl=result["test_kl_mean"],
        artifact=selected.relative_to(target).as_posix())
    for name_, key in (("mean", "train_mean"), ("median", "train_median"), ("raw", "raw_window_ridge")):
        row["forecast_vs_"+name_] = result["forecast"]["mae"]-baseline["forecast"][key]["mae"]
    row["forecast_vs_pca"] = result["forecast"]["mae"]-baseline["pca"]["forecast"]["mae"]
    for prefix, key in (("class", "classification"), ("anomaly", "anomaly")):
        for metric in ("f1", "pr_auc"):
            row[f"{prefix}_{metric}"] = result.get(key, {}).get(metric)
    for prefix, key in (("clean", "clean_reconstruction"), ("input_clean", "noisy_input_vs_clean")):
        for metric in ("mse", "mae"):
            row[f"{prefix}_{metric}"] = result.get(key, {}).get(metric)
    return row


def _execute(originals, target, protocol, started):
    rows = []
    for name, original in originals.items():
        for fold in protocol["folds"]:
            data = fold_data(original, fold)
            fractions = (.55/.85, .15/.85, .15/.85) if fold == 0 else (.70, .15, .15)
            cfg = DataConfig(window_len=48, stride=2, batch_size=128, train_frac=fractions[0],
                val_frac=fractions[1], test_frac=fractions[2], n_features=data["raw"].shape[1], seed=protocol["seeds"][0])
            fingerprint = input_fingerprint(data_arrays(data), data["feature_names"])
            group = target/name/f"fold_{fold}"
            group.mkdir(parents=True, exist_ok=True)
            baseline = _baseline(group, data, cfg, fingerprint)
            for model_name in protocol["models"]:
                model_cfg = ModelConfig(model_name=model_name, hidden_dim=32, d_model=32,
                                        tcn_levels=3, latent_dim=16, n_attn_layers=2)
                chosen_beta = 0.
                for index, seed in enumerate(protocol["seeds"]):
                    cfg.seed = seed
                    candidates = protocol["beta_grid"] if model_cfg.variational and index == 0 else [chosen_beta]
                    results = []
                    for beta in candidates:
                        exp = ExperimentConfig(cfg, model_cfg, TrainConfig(epochs=protocol["epochs"],
                            beta_target=beta, kl_anneal_epochs=5, early_stopping_patience=8,
                            seed=seed, device=protocol["device"], deterministic=protocol["deterministic"]))
                        path = group/f"{model_name}_seed{seed}_beta{beta:g}"
                        if (path/"manifest.json").exists():
                            man = verify_artifact(path, expected_config=asdict(exp), expected_fingerprint=fingerprint)
                            score = man["history"]["best_score"]
                        else:
                            if path.exists():
                                raise RuntimeError(f"Incomplete artifact: {path}; inspect before retrying")
                            print(f"TRAIN {name} fold={fold} {model_name} seed={seed} beta={beta:g}", flush=True)
                            score = train_experiment(exp, output_dir=path, **data, verbose=False, evaluate=False,
                                metadata=dict(dataset=name, fold=fold, study=str(target),
                                              beta_tuning_seed=protocol["seeds"][0]))["validation_score"]
                        results.append((score, beta, path))
                    score, chosen_beta, selected = min(results, key=lambda item: item[0])
                    if (selected/"metrics.json").exists():
                        result = verify_metrics(selected)
                    else:
                        result = save_metrics(selected, evaluate_artifact(selected, protocol["device"]))
                    row = _row(name, fold, model_name, seed, chosen_beta, score, selected, result, baseline, target)
                    rows.append(row)
                    atomic_write(target/"progress.csv", pd.DataFrame(rows).to_csv(index=False).encode("utf-8"))
                    print(f"DONE {name} fold={fold} {model_name} seed={seed} MSE={row['mse']:.5f} "
                          f"MAE={row['forecast_mae']:.5f} elapsed={time.perf_counter()-started:.0f}s", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--out")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--data-seed", type=int, default=2026)
    parser.add_argument("--datasets", nargs="+", choices=["synthetic", "aapl", "btc"], default=["synthetic", "aapl", "btc"])
    parser.add_argument("--models", nargs="+", choices=ALL_MODEL_NAMES+ABLATION_MODEL_NAMES)
    parser.add_argument("--folds", nargs="+", type=int, choices=[0, 1], default=[0, 1])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    target = args.out or str(Path(args.root)/"results"/datetime.now(timezone.utc).strftime("study_%Y%m%dT%H%M%S%fZ"))
    run_study(args.root, target, epochs=args.epochs, seeds=args.seeds, datasets=args.datasets,
              models=args.models, folds=args.folds, device=args.device, data_seed=args.data_seed,
              deterministic=args.deterministic)


if __name__ == "__main__":
    main()
