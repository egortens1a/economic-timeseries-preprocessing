"""Compare saved models with baselines and estimate paired temporal intervals."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from tsprep.experiment import load_artifact
from tsprep.provenance import (write_json, read_json, file_hash, atomic_write, save_npz, runtime_environment,
                               source_hashes, source_snapshot)
from tsprep.study import validate_completed_study
from tsprep.data_preprocessing.datasets import build_eval_loader, forecast_pairs, prepare_and_window
from tsprep.evaluation.reconstruction import collect_embeddings
from tsprep.evaluation.uncertainty import paired_block_interval


def fit_predict(xtr, ytr, xte, alpha):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(xtr, ytr).predict(xte)


def baseline_arrays(study, row, p, cfg, latent_dim):
    stored = read_json(study / row.dataset / f"fold_{row.fold}" / "baselines.json")
    train = np.asarray(p.train_windows, dtype=np.float32).reshape(len(p.train_windows), -1)
    test = np.asarray(p.test_windows, dtype=np.float32).reshape(len(p.test_windows), -1)
    pca = PCA(n_components=min(latent_dim, *train.shape), svd_solver="full").fit(train)
    ztr, zte = pca.transform(train), pca.transform(test)
    tr_ids, ytr = forecast_pairs(p, "train", cfg)
    te_ids, yte = forecast_pairs(p, "test", cfg)
    predictions = dict(train_mean=np.full_like(yte, ytr.mean()),
        train_median=np.full_like(yte, np.median(ytr)),
        last_observed=p.test_windows[te_ids, -1, cfg.target_feature],
        raw_window_ridge=fit_predict(train[tr_ids], ytr, test[te_ids], stored["forecast"]["raw_window_ridge"]["ridge_alpha"]),
        pca_ridge=fit_predict(ztr[tr_ids], ytr, zte[te_ids], stored["pca"]["forecast"]["ridge_alpha"]))
    hat = pca.inverse_transform(zte).reshape(p.test_windows.shape)
    if not np.isclose(np.mean((p.test_windows-hat)**2), stored["pca"]["mse"], rtol=1e-5, atol=1e-7):
        raise ValueError("Recomputed PCA baseline disagrees with stored baseline")
    expected = {"train_mean": stored["forecast"]["train_mean"]["mae"],
        "train_median": stored["forecast"]["train_median"]["mae"],
        "last_observed": stored["forecast"]["last_observed_transformed_value"]["mae"],
        "raw_window_ridge": stored["forecast"]["raw_window_ridge"]["mae"],
        "pca_ridge": stored["pca"]["forecast"]["mae"]}
    for name, prediction in predictions.items():
        if not np.isclose(np.abs(yte-prediction).mean(), expected[name], rtol=1e-5, atol=1e-7):
            raise ValueError(f"Recomputed forecast baseline disagrees: {name}")
    return dict(predictions=predictions, pca_hat=hat, stored=stored)


def analyze(study, device, *, output_dir=None, allow_source_mismatch=False, repeats=2000):
    study = Path(study).resolve()
    rows = validate_completed_study(study, allow_legacy=allow_source_mismatch)
    target = Path(output_dir) if output_dir else study/datetime.now(timezone.utc).strftime("analysis_%Y%m%dT%H%M%S%fZ")
    target.mkdir(parents=True, exist_ok=False)
    write_json(target/"analysis_protocol.json", dict(study=str(study), runs=len(rows), device=device,
        source_mismatch_explicitly_allowed=allow_source_mismatch, repeats=repeats, blocks=[24, 48, 96],
        min_blocks=5, source_hashes=source_hashes(), analysis_script_sha256=file_hash(__file__),
        origin_table_sha256=file_hash(study/"comparison_runs.csv"), environment=runtime_environment(device)))
    source_snapshot(target/"source.zip")
    intervals, invariance, benchmarks, details, histories = [], [], [], [], []
    losses, cache = {}, {}
    reference = rows.iloc[0]
    for row in rows.itertuples():
        artifact = study/row.artifact
        model, p, exp, manifest, arrays = load_artifact(artifact, device,
                                                     allow_source_mismatch=allow_source_mismatch)
        out = {s: collect_embeddings(model, build_eval_loader(getattr(p, s+"_windows"), exp.data.batch_size), device)
               for s in ("train", "test")}
        tr_ids, ytr = forecast_pairs(p, "train", exp.data)
        te_ids, yte = forecast_pairs(p, "test", exp.data)
        metrics = read_json(artifact/"metrics.json")
        prediction = fit_predict(out["train"]["z"][tr_ids], ytr, out["test"]["z"][te_ids],
                                 metrics["forecast"]["ridge_alpha"])
        key = (row.dataset, row.fold, exp.model.latent_dim)
        if key not in cache:
            cache[key] = baseline_arrays(study, row, p, exp.data, exp.model.latent_dim)
        base = cache[key]
        errors = ((out["test"]["x"]-out["test"]["x_hat"])**2).mean(axis=(1, 2))
        pca_errors = ((out["test"]["x"]-base["pca_hat"])**2).mean(axis=(1, 2))
        mae = np.abs(yte-prediction)
        deltas = {"forecast_mae_minus_"+name: mae-np.abs(yte-pred)
                  for name, pred in base["predictions"].items()}
        deltas["reconstruction_mse_minus_pca"] = errors-pca_errors
        detail = dict(dataset=row.dataset, fold=row.fold, model=row.model, seed=row.seed,
            mse=float(errors.mean()), forecast_mae=float(mae.mean()),
            mse_vs_pca=float((errors-pca_errors).mean()), parameters=manifest["parameters"],
            training_seconds=float(sum(manifest["history"]["epoch_time_sec"])),
            classification=metrics.get("classification"), anomaly=metrics.get("anomaly"),
            anomaly_endpoint=metrics.get("anomaly_endpoint"))
        for name, delta in deltas.items():
            detail[name] = float(delta.mean())
        clean_values = {}
        if "clean_raw" in arrays:
            clean = prepare_and_window(arrays["clean_raw"], exp.data, normalizer=p.normalizer).test_windows
            for name, loss_fn in (("mse", np.square), ("mae", np.abs)):
                model_loss = loss_fn(clean-out["test"]["x_hat"]).mean(axis=(1, 2))
                input_loss = loss_fn(clean-out["test"]["x"]).mean(axis=(1, 2))
                pca_loss = loss_fn(clean-base["pca_hat"]).mean(axis=(1, 2))
                detail["clean_"+name] = float(model_loss.mean())
                detail["input_clean_"+name] = float(input_loss.mean())
                detail["pca_clean_"+name] = float(pca_loss.mean())
                deltas[f"clean_{name}_minus_input"] = model_loss-input_loss
                deltas[f"clean_{name}_minus_pca"] = model_loss-pca_loss
                clean_values["clean_"+name+"_per_window"] = model_loss
            # Element weighting intentionally matches the original overlapping-window metrics.
            if p.is_anomaly is not None:
                mask = np.stack([p.is_anomaly[start:start+exp.data.window_len] for start in p.window_starts["test"]])
                for label, selected in (("contaminated", mask), ("normal", ~mask)):
                    for name, loss_fn in (("mse", np.square), ("mae", np.abs)):
                        point_error = loss_fn(clean-out["test"]["x_hat"]).mean(axis=2)
                        input_error = loss_fn(clean-out["test"]["x"]).mean(axis=2)
                        detail[f"clean_{name}_{label}"] = float(point_error[selected].mean()) if selected.any() else None
                        detail[f"input_clean_{name}_{label}"] = float(input_error[selected].mean()) if selected.any() else None
        losses.setdefault((row.dataset, row.fold, row.model), []).append(deltas)
        consistency = dict(artifact=row.artifact, mse_abs_difference=abs(float(errors.mean())-row.mse),
                           forecast_mae_abs_difference=abs(float(mae.mean())-row.forecast_mae))
        if max(consistency["mse_abs_difference"], consistency["forecast_mae_abs_difference"]) > 1e-5:
            raise ValueError(f"Historical reload mismatch: {consistency}")
        invariance.append(consistency)
        details.append(detail)
        h = manifest["history"]
        histories.append(dict(dataset=row.dataset, fold=row.fold, model=row.model, seed=row.seed,
            epochs=len(h["train_loss"]), best_epoch=h["best_epoch"], epoch_limit=exp.train.epochs,
            ended_at_budget=len(h["train_loss"]) == exp.train.epochs,
            best_at_last=h["best_epoch"] == len(h["train_loss"])))
        save_npz(target/(row.dataset+f"_fold{row.fold}_{row.model}_seed{row.seed}.npz"),
            dict(forecast_target=yte, forecast_prediction=prediction,
                 forecast_end_indices=p.window_starts["test"][te_ids]+exp.data.window_len-1,
                 mse_per_window=errors, pca_mse_per_window=pca_errors, **clean_values,
                 **{"forecast_baseline_"+name: pred for name, pred in base["predictions"].items()}, **deltas))
        if row.dataset == reference.dataset and row.fold == reference.fold and row.seed == reference.seed:
            batch = torch.tensor(p.test_windows[:128], dtype=torch.float32, device=device)
            with torch.no_grad():
                for _ in range(5):
                    model(batch)
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                for _ in range(20):
                    model(batch)
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
            benchmarks.append(dict(model=row.model, device=device, batch=len(batch), window_len=exp.data.window_len,
                milliseconds_per_batch=(time.perf_counter()-start)/20*1000,
                peak_allocated_bytes=torch.cuda.max_memory_allocated() if device.startswith("cuda") else None))
        print("analyzed", row.dataset, row.fold, row.model, row.seed, flush=True)
    for (dataset, fold, model), values in losses.items():
        for metric in values[0]:
            delta = np.stack([v[metric] for v in values])
            for block in (24, 48, 96):
                intervals.append(dict(dataset=dataset, fold=fold, model=model, metric=metric,
                                      **paired_block_interval(delta, block, repeats=repeats)))
    detail_frame = pd.DataFrame([{k: v for k, v in detail.items() if not isinstance(v, dict)} for detail in details])
    for filename, data in (("paired_block_intervals", intervals), ("reload_consistency", invariance),
                           ("inference_benchmark", benchmarks), ("training_diagnostics", histories)):
        atomic_write(target/(filename+".csv"), pd.DataFrame(data).to_csv(index=False).encode("utf-8"))
    atomic_write(target/"comparison_extended.csv", detail_frame.to_csv(index=False).encode("utf-8"))
    numeric = detail_frame.select_dtypes("number").columns.difference(["fold", "seed"])
    summary = detail_frame.groupby(["dataset", "model"])[numeric].agg(["mean", "std"])
    atomic_write(target/"comparison_extended_summary.csv", summary.to_csv().encode("utf-8"))
    write_json(target/"task_details.json", details)
    _plots(detail_frame, target)
    interval_frame = pd.DataFrame(intervals)
    write_json(target/"analysis_summary.json", dict(runs=len(rows),
        maximum_reload_mse_difference=max(v["mse_abs_difference"] for v in invariance),
        maximum_reload_forecast_difference=max(v["forecast_mae_abs_difference"] for v in invariance),
        interval_status_counts=interval_frame.groupby(["dataset", "status"]).size().reset_index(name="count").to_dict("records"),
        bootstrap="paired non-circular moving time blocks; resampled initialization seeds; minimum five time blocks",
        limitations=["exploratory reanalysis of already inspected test data", "conditional on selected series/protocol",
            "initialization seeds are not independent economic histories", "no multiple-comparison correction",
            "overlapping windows; folds share training history", "BTC lacks adequate blocks for all requested lengths",
            "AAPL recession evaluation has one positive episode in fold 0 and none in fold 1"]))
    write_json(target/"complete.json", dict(runs=len(rows), files={p.name: file_hash(p)
        for p in sorted(target.iterdir()) if p.is_file() and p.name != "complete.json"}))
    return target


def _plots(frame, target):
    datasets = list(frame.dataset.unique())
    fig, axes = plt.subplots(3, len(datasets), figsize=(6*len(datasets), 12), squeeze=False)
    for column, dataset in enumerate(datasets):
        sub = frame[frame.dataset == dataset]
        for index, metric in enumerate(("mse_vs_pca", "forecast_mae_minus_train_median", "forecast_mae_minus_raw_window_ridge")):
            grouped = sub.groupby("model")[metric].agg(["mean", "std"]).sort_index()
            axis = axes[index, column]
            axis.barh(grouped.index, grouped["mean"], xerr=grouped["std"].fillna(0))
            axis.axvline(0, color="black", linewidth=.8)
            titles = {"mse_vs_pca": "ΔMSE = model MSE − PCA MSE",
                "forecast_mae_minus_train_median": "ΔMAE = model MAE − train-median MAE",
                "forecast_mae_minus_raw_window_ridge": "ΔMAE = model MAE − raw-Ridge MAE"}
            axis.set_title(dataset+": "+titles[metric])
            axis.set_xlabel("Error difference: negative = model better; positive = model worse")
            axis.grid(axis="x", alpha=.2)
    fig.suptitle("Mean and SD over seeds/folds; SD is not a confidence interval")
    fig.tight_layout()
    fig.savefig(target/"baseline_comparison.png", dpi=160)
    fig.savefig(target/"baseline_comparison.svg")
    plt.close(fig)
    if "clean_mse" in frame:
        sub = frame[frame.clean_mse.notna()].copy()
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for axis, metric in zip(axes, ("mse", "mae")):
            sub["delta"] = sub["clean_"+metric]-sub["input_clean_"+metric]
            values = sub.groupby("model").delta.mean().sort_index()
            axis.barh(values.index, values)
            axis.axvline(0, color="black", linewidth=.8)
            axis.set_title("Δ"+metric.upper()+" = model clean error − observed-input clean error")
            axis.set_xlabel("Error difference: negative = model better; positive = model worse")
        fig.tight_layout()
        fig.savefig(target/"clean_tradeoff.png", dpi=160)
        fig.savefig(target/"clean_tradeoff.svg")
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study")
    parser.add_argument("--out")
    parser.add_argument("--allow-source-mismatch", action="store_true")
    parser.add_argument("--repeats", type=int, default=2000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        print(analyze(args.study, args.device, output_dir=args.out,
                      allow_source_mismatch=args.allow_source_mismatch, repeats=args.repeats))
