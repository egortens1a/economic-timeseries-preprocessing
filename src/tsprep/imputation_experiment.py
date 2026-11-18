"""Обучение и сравнение методов заполнения пропусков по NPZ или исходному ZIP NN5.

CLI: python -m tsprep.imputation_experiment --data dataset.zip --out results/my_run
"""
import argparse
from dataclasses import asdict, replace
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import torch

from tsprep.config import ImputationProtocol
from tsprep.imputation import data as d, baselines as f, training as n
from tsprep.imputation.evaluation import decide, fit_forecast, forecast_features, forecast_predict, plot
from tsprep.mlcore.trainer import seed_everything
from tsprep.provenance import (file_hash, runtime_environment, save_npz,
                               source_hashes, source_snapshot, write_json)


def serial(value):
    if isinstance(value, dict):
        return {key: serial(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def metric(name, blocks):
    count = int(blocks["count"].sum())
    return dict(method=name, scored_points=count, scaled_mae=float(blocks["scaled"].sum()/count),
                native_mae=float(blocks["absolute"].sum()/count),
                native_rmse=float(np.sqrt(blocks["squared"].sum()/count)))


def fit_baselines(train, val, scaler, out, p):
    fitted, predictions, candidates = {}, {}, []
    for method in f.METHODS[:4]:
        predictions[method] = f.predict(method, val, scaler, fitted)
    for rank in p.pca_ranks:
        if rank > min(train["windows"].shape):
            continue
        state = f.fit_pca(train, scaler, rank)
        prediction = f.pca_fill(val, scaler, state)
        score = d.error_blocks(prediction, val, scaler)["score"]
        candidates.append(dict(method="pca", rank=rank, score=score))
        if "pca" not in fitted or score < fitted["pca_score"]:
            fitted.update(pca=state, pca_score=score)
            predictions["pca"] = prediction
   
    if "pca" not in fitted:
        raise ValueError("no PCA rank fits the training matrix; choose a smaller pca_ranks value")
   
    for alpha in p.ridge_alphas:
        state = f.fit_ridge(train, scaler, alpha, p.missing_fraction)
        prediction = f.ridge_fill(val, scaler, state)
        score = d.error_blocks(prediction, val, scaler)["score"]
        candidates.append(dict(method="ridge_gap", alpha=alpha, score=score))
        if "ridge_gap" not in fitted or score < fitted["ridge_score"]:
            fitted.update(ridge_gap=state, ridge_score=score)
            predictions["ridge_gap"] = prediction
   
    scores = {name: d.error_blocks(value, val, scaler)["score"] for name, value in predictions.items()}
    best = min(f.METHODS, key=lambda name: scores[name])
    write_json(out/"baseline_selection.json", serial(dict(fitted=fitted, scores=scores, best=best, candidates=candidates)))
    print("BASELINES", scores, "BEST", best, flush=True)
    return fitted, predictions, best


def fit_networks(train, val, scaler, fitted, best, validation_base, device, out, p):
    models, candidates, training = {}, [], []
    for lr in p.learning_rates:
        for seed in p.initialization_seeds:
            model, history = n.fit(train, val, scaler, fitted, best, validation_base, lr, seed, device, p)
            folder = out/"models"/f"lr{lr}_seed{seed}"
            folder.mkdir(parents=True)
            torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, folder/"model.pt")
            write_json(folder/"training.json", history)
            models[lr, seed] = model
            training.append({key: value for key, value in history.items() if key != "history"})
            print(f"FIT lr={lr} seed={seed} best_epoch={history['best_epoch']} val={history['validation_score']:.6f} seconds={history['seconds']:.2f}", flush=True)
  
        for shrink in p.shrinkages:
            scores = [d.error_blocks(n.predict(models[lr, seed], val, scaler, validation_base, device, shrink), val, scaler)["score"] for seed in p.initialization_seeds]
            candidates.append(dict(learning_rate=lr, shrinkage=shrink, score=float(np.mean(scores)), seed_scores=scores))
 
    chosen = min(candidates, key=lambda item: item["score"])
    selected = {seed: models[chosen["learning_rate"], seed] for seed in p.initialization_seeds}
    pd.DataFrame(training).to_csv(out/"training_summary.csv", index=False)
    write_json(out/"neural_selection.json", dict(selected=chosen, candidates=candidates,
        model_paths={str(seed): f"models/lr{chosen['learning_rate']}_seed{seed}/model.pt" for seed in p.initialization_seeds}))
  
    return selected, chosen


def filled(method, frame, scaler, fitted, best, models, selected, device, base):
    if method.startswith("neural_"):
        seed = int(method.split("_")[1])
        return n.predict(models[seed], frame, scaler, base, device, selected["shrinkage"])
    return base if method == best else f.predict(method, frame, scaler, fitted)


def fit_forecasts(raw, normalized, scaler, fitted, best, models, selected, device, methods, out, p):
    train = d.forecast_windows(raw, normalized, 0, p.train_stop, p=p)
    val = d.forecast_windows(raw, normalized, p.train_stop, p.val_stop, p=p)
    train_base, val_base = (f.predict(best, frame, scaler, fitted) for frame in (train, val))
    probes = {}
    for method in methods:
        tx = forecast_features(filled(method, train, scaler, fitted, best, models, selected, device, train_base), train)
        vx = forecast_features(filled(method, val, scaler, fitted, best, models, selected, device, val_base), val)
        probes[method] = fit_forecast(tx, train, vx, val, scaler, p.ridge_alphas)
        print(f"FORECAST selected {method} alpha={probes[method]['alpha']} val={probes[method]['validation_score']:.6f}", flush=True)
   
    write_json(out/"forecast_probes.json", probes)
    return probes


def evaluate_forecasts(raw, normalized, scaler, fitted, best, models, selected, device, methods, probes, masks, out, p):
    rows = []
    for condition, extra_masks in (("natural_missing", [None]), (f"additional{100*p.missing_fraction:g}pct_missing", masks)):
        for draw, extra in enumerate(extra_masks):
            frame = d.forecast_windows(raw, normalized, p.val_stop, raw.shape[1], extra, p)
            base = f.predict(best, frame, scaler, fitted)
            saved = dict(target=frame["target"], atm=frame["atm"], target_index=frame["target_index"])
            all_errors = {}
            for method in methods:
                value = filled(method, frame, scaler, fitted, best, models, selected, device, base)
                prediction = forecast_predict(probes[method], forecast_features(value, frame), frame, scaler)
                saved[method] = prediction
                errors = np.abs(prediction-frame["target"])
                all_errors[method] = errors
                rows.append(dict(condition=condition, draw=draw, method=method, native_mae=float(errors.mean()),
                                 scaled_mae=float((errors/scaler["mase"][frame["atm"]]).mean()), count=len(errors)))
    
            for label, position in (("last", -1), ("seasonal", -p.period)):
                prediction = np.maximum(np.expm1(base[:, position]*scaler["std"][frame["atm"]]+scaler["mean"][frame["atm"]]), 0.)
                saved[label] = prediction
                error = np.abs(prediction-frame["target"])
                rows.append(dict(condition=condition, draw=draw, method=label, native_mae=float(error.mean()),
                                 scaled_mae=float((error/scaler["mase"][frame["atm"]]).mean()), count=len(error)))
  
            error = np.mean([all_errors[f"neural_{seed}"] for seed in p.initialization_seeds], axis=0)
            rows.append(dict(condition=condition, draw=draw, method="neural_mean_error", native_mae=float(error.mean()),
                             scaled_mae=float((error/scaler["mase"][frame["atm"]]).mean()), count=len(error)))
            save_npz(out/"test"/f"forecast_{condition}_{draw}.npz", saved)
 
    pd.DataFrame(rows).to_csv(out/"forecast_metrics.csv", index=False)
    pd.DataFrame(rows).groupby(["condition", "method"], as_index=False)[["native_mae", "scaled_mae", "count"]].mean().to_csv(out/"forecast_summary.csv", index=False)


def run_experiment(data_path, out, p=ImputationProtocol()):
    """Обучает методы заполнения и сохраняет результаты в новую папку."""
    data_path, out = Path(data_path), Path(out)
    panel = d.load_panel(data_path)
    raw = panel["values"]
    p.validate(raw.shape[1])
    # Validate train-only statistics before creating an output directory.
    scaler, normalized = d.fit_scaler(raw, p)
    out.mkdir(parents=True, exist_ok=False)
    begin = time.perf_counter()
    with threadpool_limits(limits=p.threads):
        torch.set_num_threads(p.threads)
        seed_everything(111, deterministic=True)
        for folder in ("inputs", "models", "validation", "test"):
            (out/folder).mkdir()
 
        shutil.copy2(data_path, out/"inputs"/("original"+data_path.suffix.lower()))
        save_npz(out/"inputs/panel.npz", panel)
        hashes = source_hashes()
        protocol = dict(**asdict(p), length=raw.shape[1], series=raw.shape[0],
            protocol_id="masked_imputation_v1",
            primary="recovery of additionally hidden known values; natural NaNs have no labels",
            availability="entire past window at forecast origin; no observations after origin",
            training="masked native MAE / train seasonal MAE; known cells copied exactly",
            selection="baseline, checkpoint, learning rate and shrinkage selected on validation only",
            inference="mean errors over initialization seeds; no prediction ensemble",
            forecast="separate secondary H1 Ridge check, not the original NN5 H56 competition")
  
        write_json(out/"protocol.json", protocol)
        plan = dict(protocol=protocol, device=p.device, package_sources=hashes, input_name=data_path.name,
                    input_sha256=file_hash(data_path),
                    inputs={path.name: file_hash(path) for path in sorted((out/"inputs").iterdir())})
 
        write_json(out/"plan.json", plan)
        source_snapshot(out/"source.zip")
        write_json(out/"environment.json", runtime_environment(p.device))
        save_npz(out/"scaler.npz", scaler)
        print("PLAN FROZEN", file_hash(out/"plan.json"), flush=True)
        train = d.train_windows(normalized, p)
        train["truth_raw"] = np.stack([raw[s, a:a+p.window] for s, a in zip(train["atm"], train["start"])])
        val_masks = d.additional_masks(normalized, scaler, p.train_stop, p.val_stop,
                                       p.validation_mask_seeds, p.missing_fraction)
        validation = d.evaluation_windows(raw, normalized, val_masks, p.train_stop, p.val_stop, p)
        save_npz(out/"validation/masks.npz", dict(extra=val_masks))
        save_npz(out/"validation/windows.npz", validation)
        fitted, val_predictions, best = fit_baselines(train, validation, scaler, out, p)
        models, selected = fit_networks(train, validation, scaler, fitted, best, val_predictions[best], p.device, out, p)
        methods = list(f.METHODS)+[f"neural_{seed}" for seed in p.initialization_seeds]
        for seed in p.initialization_seeds:
            val_predictions[f"neural_{seed}"] = n.predict(models[seed], validation, scaler, val_predictions[best], p.device, selected["shrinkage"])
        save_npz(out/"validation/predictions.npz", val_predictions)
        probes = fit_forecasts(raw, normalized, scaler, fitted, best, models, selected, p.device, methods, out, p)
        frozen = [path for folder in ("inputs", "models", "validation") for path in sorted((out/folder).rglob("*")) if path.is_file()]
        frozen += [out/name for name in ("protocol.json", "plan.json", "source.zip", "environment.json", "scaler.npz", "baseline_selection.json", "neural_selection.json", "forecast_probes.json", "training_summary.csv")]
        selection = dict(baseline=best, neural=selected, methods=methods,
                         files={path.relative_to(out).as_posix(): file_hash(path) for path in frozen})
        write_json(out/"selection.json", selection)
        selection_hash = file_hash(out/"selection.json")
        print("SELECTION FROZEN", best, selected, flush=True)
        test_masks = d.additional_masks(normalized, scaler, p.val_stop, raw.shape[1],
                                        p.test_mask_seeds, p.missing_fraction)
        test = d.evaluation_windows(raw, normalized, test_masks, p.val_stop, raw.shape[1], p)
        save_npz(out/"test/masks.npz", dict(extra=test_masks))
        save_npz(out/"test/windows.npz", test)
        base = f.predict(best, test, scaler, fitted)
        predictions = {name: filled(name, test, scaler, fitted, best, models, selected, p.device, base) for name in methods}
        save_npz(out/"test/predictions.npz", predictions)
        blocks = {name: d.error_blocks(value, test, scaler) for name, value in predictions.items()}
        summaries = [metric(name, value) for name, value in blocks.items()]
        average = {key: np.mean([blocks[f"neural_{seed}"][key] for seed in p.initialization_seeds], axis=0) for key in ("scaled", "absolute", "squared", "count")}
        summaries.append(metric("neural_mean_error", average))
        pd.DataFrame(summaries).to_csv(out/"imputation_summary.csv", index=False)
        save_npz(out/"test/error_blocks.npz", {name+"__"+key: value[key] for name, value in blocks.items() for key in ("scaled", "absolute", "squared", "count")})
        decision, bootstrap = decide(blocks, best, p)
        write_json(out/"decision.json", decision)
        save_npz(out/"bootstrap_indices.npz", bootstrap)
        evaluate_forecasts(raw, normalized, scaler, fitted, best, models, selected, p.device, methods, probes, test_masks, out, p)
        plot(out, panel, test, predictions, best, p)
        if source_hashes() != hashes or file_hash(out/"selection.json") != selection_hash:
            raise RuntimeError("source or selection changed during the experiment")
        for name, digest in selection["files"].items():
            if file_hash(out/name) != digest:
                raise RuntimeError(f"frozen input changed: {name}")
        files = {path.relative_to(out).as_posix(): file_hash(path) for path in sorted(out.rglob("*")) if path.is_file()}
        write_json(out/"complete.json", dict(status="complete", seconds=time.perf_counter()-begin,
                   joint_success=decision["joint_success"], selection_sha256=selection_hash, files=files))
        print("PRIMARY", decision, flush=True)
        print("SAVED", out, flush=True)
        return decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="NPZ with values[series,time], or original daily NN5 ZIP")
    parser.add_argument("--out", required=True, help="Новая папка результатов")
    defaults = ImputationProtocol()
    for name in ("window", "period", "train_stop", "val_stop", "epochs", "batch_size", "bootstrap_repeats", "threads"):
        parser.add_argument("--"+name.replace("_", "-"), type=int, default=getattr(defaults, name))
    for name in ("initialization_seeds", "validation_mask_seeds", "test_mask_seeds", "pca_ranks"):
        parser.add_argument("--"+name.replace("_", "-"), type=int, nargs="+", default=getattr(defaults, name))
    for name in ("learning_rates", "ridge_alphas", "shrinkages"):
        parser.add_argument("--"+name.replace("_", "-"), type=float, nargs="+", default=getattr(defaults, name))
    parser.add_argument("--missing-fraction", type=float, default=defaults.missing_fraction)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke", action="store_true", help="Проверка запуска: одна эпоха и одна инициализация")
    args = vars(parser.parse_args())
    data_path, out, smoke = args.pop("data"), args.pop("out"), args.pop("smoke")
    for key in ("initialization_seeds", "validation_mask_seeds", "test_mask_seeds", "pca_ranks", "learning_rates", "ridge_alphas", "shrinkages"):
        args[key] = tuple(args[key])
    p = ImputationProtocol(**args)
    if smoke:
        p = replace(p, epochs=1, initialization_seeds=p.initialization_seeds[:1],
                    learning_rates=p.learning_rates[:1], bootstrap_repeats=100)
    run_experiment(data_path, out, p)


if __name__ == "__main__":
    main()
