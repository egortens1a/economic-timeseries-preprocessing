"""Признаки из исправленных окон и прогноз с помощью Ridge."""
import numpy as np
from tsprep.denoising.baselines import baseline_predictions, delayed_pair_values
from tsprep.denoising.data import window_inputs, denoising_indices
from tsprep.denoising.training import predict
from tsprep.evaluation.downstream import fit_probe, predict_probe


def forecast_features(windows, models, baseline, pca, p):
    values = baseline_predictions(windows, baseline["mean"], pca,
        baseline["choices"]["causal_median"]["setting"], baseline["choices"]["causal_hampel"]["setting"],
        endpoint_offset=p.correction_delay)
    # Фильтр Хампеля применяем к каждой точке с учётом только доступной на ней истории.
    width, threshold = baseline["choices"]["causal_hampel"]["setting"]
    flat = windows[:, :, 0]
    h = flat.copy()
    positions = range(p.window-p.forecast_features-p.correction_delay, p.window-p.correction_delay)
    for t in positions:
        previous = flat[:, t-width:t]
        center = np.median(previous, axis=1)
        mad = np.median(np.abs(previous-center[:, None]), axis=1)
        h[:, t] = np.where(np.abs(flat[:, t]-center) > threshold*np.maximum(1.4826*mad, 1e-6), center, flat[:, t])
    values["causal_hampel"] = h
    center, scale = baseline["global_center"], baseline["global_scale"]
    cutoff = baseline["choices"]["global_mad"]["setting"] * scale
    values["global_mad"] = np.where(np.abs(flat-center) > cutoff, center, flat)
    if p.correction_delay:
        centered = flat.copy()
        for t in positions:
            centered[:, t] = np.median(flat[:, t-1:t+2], axis=1)
        values["centered_median"] = centered
        values["delayed_pair"] = delayed_pair_values(flat, baseline["choices"]["delayed_pair"]["setting"], positions)
        # У фильтров индекс задаёт исправляемую точку, у нейросети момент доступности данных.
        values = {name: np.pad(a[:, :-1], ((0, 0), (1, 0))) for name, a in values.items()}
    
    for seed, model in models.items():
        values[f"residual_tcn_{seed}"] = predict(model, windows, p.device, p.batch_size)[:, :, 0]
    
    return {name: a[:, -p.forecast_features:] for name, a in values.items()}


def fit_forecasts(development, models, baseline, pca, p):
    data = {}
    for split in ("train", "val"):
        features, targets = [], []
        for t in development:
            segment = t[split]
            windows = window_inputs(segment, p, p.training_stride)
            _, availability = denoising_indices(segment, p, p.training_stride)
            valid = availability+p.forecast_horizon < len(segment["clean"])
            extracted = forecast_features(windows[valid], models, baseline, pca, p)
            if p.correction_delay:
                extracted["raw_available"] = windows[valid, -p.forecast_features:, 0]
            features.append(extracted)
            targets.append(segment["clean"][availability[valid]+p.forecast_horizon, 0])
        
        data[split] = ({name: np.concatenate([f[name] for f in features]) for name in features[0]}, np.concatenate(targets))
    
    probes = {name: fit_probe(x, data["train"][1], data["val"][0][name], data["val"][1])
              for name, x in data["train"][0].items()}
    return dict(probes=probes, mean=float(data["train"][1].mean()),
                median=float(np.median(data["train"][1])),
                target=(f"clean_log_return_at_availability_plus_{p.forecast_horizon}" if p.correction_delay else "clean_log_return_at_endpoint_plus_7"), horizon=p.forecast_horizon,
                feature_count=p.forecast_features, tuning="Ridge alpha on validation only")


def embedding_features(segment, p, models, fitted, severity=1):
    from tsprep.data_preprocessing.datasets import TensorWindowLoader
    from tsprep.denoising.baselines import representation_baseline_features
    from tsprep.denoising.data import representation_windows
    from tsprep.evaluation.reconstruction import collect_embeddings
    windows, ends, target = representation_windows(segment, p, severity=severity)
    clean = window_inputs(segment, p, clean_input=True)[:len(windows)]
    features = representation_baseline_features(windows, clean, fitted)
    reconstruction = dict(raw=windows[:, -1, 0], clean_oracle=clean[:, -1, 0])
    for name in ("pca", "mad_pca"):
        reconstruction[name] = features[name] @ fitted[name].components_[:, -1] + fitted[name].mean_[-1]
    
    for name, model in models.items():
        outputs = collect_embeddings(model, TensorWindowLoader(windows, p.batch_size, p.device), p.device)
        features[name] = outputs["z"]
        reconstruction[name] = outputs["x_hat"][:, -1, 0]
    
    return features, reconstruction, ends, target

