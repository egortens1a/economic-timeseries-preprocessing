"""Оценка выбранных моделей на новых траекториях, сохранение таблиц и графиков."""
import numpy as np
import pandas as pd
from tsprep.data_preprocessing.denoising import generate_denoising_series
from tsprep.denoising.data import series_arrays, trajectory_segments, window_inputs, denoising_indices
from tsprep.denoising.forecast import forecast_features, predict_probe
from tsprep.denoising.metrics import confirmation_decision, endpoint_metrics
from tsprep.provenance import atomic_write, input_fingerprint, save_npz, write_json


def write_csv(path, frame):
    atomic_write(path, frame.to_csv(index=False).encode("utf-8"))


def save_figures(out, frame, p):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    means = frame.groupby(["family", "data_seed", "model"])[["all_mae", "normal_mse"]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric in zip(axes, ("all_mae", "normal_mse")):
        values = means[metric].unstack("model")
        ratios = values.div(values["observed_input"], axis=0).mean().sort_values()
        ax.barh(ratios.index, ratios.values)
        ax.axvline(1, color="black", linestyle="--")
        ax.set_xlabel(f"{metric} / observed input (lower is better)")
    fig.tight_layout()
    fig.savefig(out/"quality_ratios.png", dpi=160)
    plt.close(fig)
    for family in p.families:
        with np.load(out/"predictions"/f"{family}_{p.confirmation_seeds[0]}.npz") as a:
            hits = np.flatnonzero(a["anomaly"])
            center = int(hits[len(hits)//2]) if len(hits) else 150
            lo, hi = max(0, center-30), min(len(a["clean"]), center+40)
            fig, ax = plt.subplots(figsize=(11, 3))
            for name in ("observed_input", "clean", "residual_tcn_111"):
                ax.plot(a["endpoint_index"][lo:hi], a[name][lo:hi], label=name)
            ax.set_title(f"{family}: fixed first confirmation trajectory")
            ax.set_ylabel("normalized log return")
            ax.legend()
            fig.tight_layout()
            fig.savefig(out/f"example_{family}.png", dpi=160)
            plt.close(fig)


def evaluate_confirmation(p, normalizer, models, baseline, pca, forecasts, out, fingerprints):
    rows, forecast_rows, pristine_rows = [], [], []
    (out/"predictions").mkdir()
    for family in p.families:
        for seed in p.confirmation_seeds:
            s = generate_denoising_series(family, seed, p.length)
            arrays = series_arrays(s)
            key = f"confirmation_{family}_{seed}"
            save_npz(out/"inputs"/f"{key}.npz", arrays)
            fingerprints[key] = input_fingerprint(arrays, ["price"])
            segment = trajectory_segments(s, p, normalizer, development=False)["test"]
            windows = window_inputs(segment, p)
            predictions = forecast_features(windows, models, baseline, pca, p)
            predictions["ordinary_residual"] = predictions["observed_input"]
            ends, availability = denoising_indices(segment, p)
            clean, mask = segment["clean"][ends, 0], segment["anomaly"][ends]
            target = segment["clean"][availability[:-p.forecast_horizon]+p.forecast_horizon, 0]
            saved = dict(clean=clean, anomaly=mask, endpoint_index=ends)
            if p.correction_delay:
                saved.update(availability_index=availability, forecast_origin_index=availability[:-p.forecast_horizon])
            
            for name, features in predictions.items():
                prediction = features[:, -1]
                model_name = "residual_tcn" if name.startswith("residual_tcn_") else name
                init = int(name.rsplit("_", 1)[1]) if model_name == "residual_tcn" else -1
                rows.append(dict(family=family, data_seed=seed, model=model_name, initialization_seed=init,
                                 **endpoint_metrics(prediction, clean, mask,
                                     large_threshold=baseline.get("large_move_thresholds", {}).get(family))))
                
                saved[name] = prediction
                if name in forecasts["probes"]:
                    pred = predict_probe(forecasts["probes"][name], features[:-p.forecast_horizon])
                    forecast_rows.append(dict(family=family, data_seed=seed, model=model_name, initialization_seed=init,
                        mae=float(np.abs(pred-target).mean()), mse=float(np.square(pred-target).mean()), count=len(target)))
                    
                    saved[f"forecast_{name}"] = pred
            
            if p.correction_delay:
                pred = predict_probe(forecasts["probes"]["raw_available"], windows[:-p.forecast_horizon, -p.forecast_features:, 0])
                forecast_rows.append(dict(family=family, data_seed=seed, model="raw_available", initialization_seed=-1,
                    mae=float(np.abs(pred-target).mean()), mse=float(np.square(pred-target).mean()), count=len(target)))
                saved["forecast_raw_available"] = pred
            
            saved["forecast_target"] = target
            for name in ("mean", "median"):
                forecast_rows.append(dict(family=family, data_seed=seed, model=f"forecast_{name}", initialization_seed=-1,
                    mae=float(np.abs(target-forecasts[name]).mean()), mse=float(np.square(target-forecasts[name]).mean()), count=len(target)))
            
            save_npz(out/"predictions"/f"{family}_{seed}.npz", saved)
            # проверяем те же траектории без ошибок наблюдения
            pristine = window_inputs(segment, p, clean_input=True)
            pristine_features = forecast_features(pristine, models, baseline, pca, p)
            for name, features in pristine_features.items():
                pristine_rows.append(dict(family=family, data_seed=seed, model=name,
                    mae=float(np.abs(features[:, -1]-clean).mean()), mse=float(np.square(features[:, -1]-clean).mean())))
            
            print(f"confirmation {family} seed={seed} saved", flush=True)
            
    frame, forecast_frame = pd.DataFrame(rows), pd.DataFrame(forecast_rows)
    write_csv(out/"endpoint_metrics.csv", frame)
    write_csv(out/"forecast_metrics.csv", forecast_frame)
    write_csv(out/"pristine_metrics.csv", pd.DataFrame(pristine_rows))
    decision = confirmation_decision(frame, p, baseline["reference_baseline"])
    write_json(out/"decision.json", decision)
    write_json(out/"input_fingerprints.json", fingerprints)
    averaged = frame.groupby(["family", "data_seed", "model"], as_index=False)[["all_mae", "all_mse", "normal_mae", "normal_mse", "contaminated_mae", "contaminated_mse"]].mean()
    write_csv(out/"summary_by_family.csv", averaged.groupby(["family", "model"], as_index=False).mean(numeric_only=True).drop(columns="data_seed"))
    write_csv(out/"summary.csv", averaged.groupby("model", as_index=False).mean(numeric_only=True).drop(columns="data_seed"))
    write_csv(out/"forecast_summary.csv", forecast_frame.groupby(["family", "data_seed", "model"], as_index=False)[["mae", "mse"]].mean().groupby("model", as_index=False).mean(numeric_only=True).drop(columns="data_seed"))
    save_figures(out, frame, p)
    if p.transfer_checks:
        evaluate_transfer(p, normalizer, models, baseline, pca, out, fingerprints)
        write_json(out/"input_fingerprints.json", fingerprints)
    
    return decision


TRANSFER_CONDITIONS = {
    "clean": dict(contaminated=False),
    "small_frequent": dict(impulse_probability=.03, impulse_bounds=(.01, .03)),
    "large_rare": dict(impulse_probability=.005, impulse_bounds=(.08, .12)),
    "clusters": dict(impulse_probability=.01, impulse_bounds=(.04, .08), impulse_duration=4),
}


def transfer_conditions(p):
    conditions = dict(TRANSFER_CONDITIONS)
    for scale in p.volatility_scales:
        if scale != 1.:
            conditions[f"volatility_{scale:g}"] = dict(volatility_scale=scale)
            conditions[f"clean_volatility_{scale:g}"] = dict(volatility_scale=scale, contaminated=False)
    
    return conditions


def evaluate_transfer(p, normalizer, models, baseline, pca, out, fingerprints):
    """Оценивает фиксированные модели при разных ошибках наблюдения на одинаковой чистой динамике."""
    rows = []
    (out/"transfer_predictions").mkdir()
    for family in p.families:
        threshold = baseline["large_move_thresholds"][family]
        for seed in p.confirmation_seeds:
            for condition, settings in transfer_conditions(p).items():
                series = generate_denoising_series(family, seed, p.length, **settings)
                arrays = series_arrays(series)
                key = f"transfer_{condition}_{family}_{seed}"
                save_npz(out/"inputs"/f"{key}.npz", arrays)
                fingerprints[key] = input_fingerprint(arrays, ["price"])
                segment = trajectory_segments(series, p, normalizer, development=False)["test"]
                values = forecast_features(window_inputs(segment, p), models, baseline, pca, p)
                ends, availability = denoising_indices(segment, p)
                clean, mask = segment["clean"][ends, 0], segment["anomaly"][ends]
                saved = dict(clean=clean, anomaly=mask, endpoint_index=ends)
                if p.correction_delay:
                    saved["availability_index"] = availability
                for name, features in values.items():
                    model = "residual_tcn" if name.startswith("residual_tcn_") else name
                    init = int(name.rsplit("_", 1)[1]) if model == "residual_tcn" else -1
                    saved[name] = features[:, -1]
                    rows.append(dict(condition=condition, family=family, data_seed=seed, model=model,
                        initialization_seed=init, **endpoint_metrics(saved[name], clean, mask, large_threshold=threshold)))
                
                save_npz(out/"transfer_predictions"/f"{condition}_{family}_{seed}.npz", saved)
            
            print(f"transfer {family} seed={seed} saved", flush=True)
    
    frame = pd.DataFrame(rows)
    write_csv(out/"transfer_metrics.csv", frame)
    metrics = [c for c in frame if c.endswith(("_mae", "_mse"))]
    averaged = frame.groupby(["condition", "family", "data_seed", "model"], as_index=False)[metrics].mean()
    write_csv(out/"transfer_summary_by_family.csv", averaged.groupby(["condition", "family", "model"], as_index=False)[metrics].mean())


def evaluate_embeddings(p, normalizer, models, baseline, selection, out, fingerprints):
    from tsprep.denoising.forecast import embedding_features
    from tsprep.denoising.metrics import representation_decision
    from tsprep.evaluation.downstream import regression_report
    from tsprep.evaluation.latent_space import latent_diagnostics
    rows, recon_rows, latent_rows = [], [], []
    (out/"predictions").mkdir()
    probes = dict(baseline["probes"], **{name: r["probe"] for name, r in selection["final_models"].items()})
    for family in p.families:
        for seed in p.confirmation_seeds:
            series = generate_denoising_series(family, seed, p.length)
            arrays = series_arrays(series)
            key = f"confirmation_{family}_{seed}"
            save_npz(out/"inputs"/f"{key}.npz", arrays)
            fingerprints[key] = input_fingerprint(arrays, ["price"])
            segment = trajectory_segments(series, p, normalizer, development=False)["test"]
            for severity in (0, 1, 2):
                features, reconstruction, ends, target = embedding_features(segment, p, models, baseline, severity)
                saved = dict(target=target, endpoint_index=ends, clean_endpoint=segment["clean"][ends, 0])
                mask = segment["anomaly"][ends] if severity else np.zeros(len(ends), bool)
                for name in (*features, "mean", "median"):
                    method, init = name.rsplit("_seed", 1) if "_seed" in name else (name, -1)
                    identity = dict(family=family, data_seed=seed, severity=severity, model=method,
                                    initialization_seed=int(init))
                    pred = predict_probe(probes[name], features[name]) if name in features else np.full(len(target), baseline[name])
                    rows.append(dict(**identity, **regression_report(target, pred),
                                     mse=float(np.square(pred-target).mean()), count=len(target)))
                    saved[f"prediction_{name}"] = pred
                    if name in reconstruction:
                        recon_rows.append(dict(**identity, **endpoint_metrics(reconstruction[name], saved["clean_endpoint"], mask)))
                    if name in features:
                        latent_rows.append(dict(**identity, dimension=features[name].shape[1], **latent_diagnostics(features[name])))
                        if features[name].shape[1] == p.latent_dim:
                            saved[f"features_{name}"] = features[name]
                
                saved["anomaly"] = mask
                save_npz(out/"predictions"/f"{family}_{seed}_severity{severity}.npz", saved)
            
            print(f"CONFIRMED {family} seed={seed}: conditions 0/1/2", flush=True)
    
    frame = pd.DataFrame(rows)
    decision = representation_decision(frame, p)
    write_csv(out/"forecast_metrics.csv", frame)
    write_csv(out/"reconstruction_metrics.csv", pd.DataFrame(recon_rows))
    write_csv(out/"latent_metrics.csv", pd.DataFrame(latent_rows))
    averaged = frame.groupby(["family", "data_seed", "severity", "model"], as_index=False)[["mae", "mse", "r2"]].mean()
    write_csv(out/"summary.csv", averaged.groupby(["severity", "model"], as_index=False)[["mae", "mse", "r2"]].mean())
    write_csv(out/"summary_by_family.csv", averaged.groupby(["family", "severity", "model"], as_index=False)[["mae", "mse", "r2"]].mean())
    write_json(out/"decision.json", decision)
    write_json(out/"input_fingerprints.json", fingerprints)
    save_embedding_figures(out, frame)
    return decision


def save_embedding_figures(out, frame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    means = frame.groupby(["family", "data_seed", "severity", "model"])["mae"].mean()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    standard = means.xs(1, level="severity").unstack("model")
    ratio = standard.div(standard["raw"], axis=0).mean().sort_values()
    axes[0].barh(ratio.index, ratio.values)
    axes[0].axvline(1, color="black", linestyle="--")
    axes[0].set_xlabel("Forecast MAE / raw Ridge")
    for name in ("raw", "pca", "mad_pca", "ordinary", "denoising"):
        trend = means.xs(name, level="model").groupby("severity").mean()
        axes[1].plot(trend.index, trend.values, marker="o", label=name)
    
    axes[1].set(xlabel="Observation error multiplier", ylabel="Clean H=7 forecast MAE", xticks=[0, 1, 2])
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out/"embedding_quality.png", dpi=160)
    plt.close(fig)
