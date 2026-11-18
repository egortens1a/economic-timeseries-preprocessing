"""Простые методы сравнения; параметры выбираются на проверочной выборке."""
import numpy as np
from sklearn.decomposition import PCA
from tsprep.denoising.data import window_inputs, denoising_indices
from tsprep.denoising.metrics import endpoint_metrics, relative_scores, validation_guard
from tsprep.provenance import save_npz, write_json
from tsprep.denoising.data import pooled_representation_data
from tsprep.evaluation.downstream import fit_probe


def baseline_predictions(windows, mean, pca, median_width, hampel, *, endpoint_offset=0):
    flat = windows[:, :, 0]
    width, threshold = hampel
    position = flat.shape[1]-1-endpoint_offset
    previous = flat[:, position-width:position]
    center = np.median(previous, axis=1)
    mad = np.median(np.abs(previous-center[:, None]), axis=1)
    endpoint = flat[:, position]
    replaced = np.where(np.abs(endpoint-center) > threshold*np.maximum(1.4826*mad, 1e-6), center, endpoint)
    hampel_values = flat.copy()
    hampel_values[:, position] = replaced
    return dict(observed_input=flat, train_mean=np.full_like(flat, mean),
                pca=pca.inverse_transform(pca.transform(flat)),
                causal_median=np.stack([np.median(windows[:, max(0, t-median_width+1):t+1, 0], axis=1)
                                        for t in range(windows.shape[1])], axis=1),
                causal_hampel=hampel_values)


def delayed_pair_values(flat, threshold, positions):
    """Correct opposite large adjacent returns using only one subsequent observation."""
    positions = np.asarray(list(positions), int)
    if not len(positions) or positions.min() < 16 or positions.max()+1 >= flat.shape[1]:
        raise ValueError("pair filter needs 16 past values and one subsequent value")
    previous = np.stack([flat[:, t-16:t] for t in positions], axis=1)
    center = np.median(previous, axis=-1)
    scale = np.maximum(1.4826*np.median(np.abs(previous-center[..., None]), axis=-1), 1e-6)
    first, second = flat[:, positions]-center, flat[:, positions+1]-center
    detected = ((first*second < 0) & (np.minimum(np.abs(first), np.abs(second)) > threshold*scale)
                & (np.abs(first+second) <= 3*scale))
    values = flat.copy()
    values[:, positions] = np.where(detected, center+(first+second)/2, flat[:, positions])
    return values


def tune_baselines(development, p, out):
    train = np.concatenate([window_inputs(t["train"], p, p.training_stride)[:, :, 0] for t in development])
    train_values = np.concatenate([t["train"]["observed"] for t in development])
    mean = float(train_values.mean())
    global_center = float(np.median(train_values))
    global_scale = max(float(1.4826*np.median(np.abs(train_values-global_center))), 1e-6)
    position = p.window-1-p.correction_delay
    scores = []
    def score(name, predictor, setting):
        values = []
        for t in development:
            segment = t["val"]
            windows = window_inputs(segment, p)
            ends, _ = denoising_indices(segment, p)
            target, mask = segment["clean"][ends, 0], segment["anomaly"][ends]
            reference = endpoint_metrics(windows[:, position, 0], target, mask)
            values.append(relative_scores(endpoint_metrics(predictor(windows), target, mask), reference))
        mae, guard = np.mean(values, axis=0)
        guard = validation_guard(values, development, p)
        scores.append(dict(model=name, setting=setting, mae_ratio=mae, normal_mse_ratio=guard))
    pcas = {}
    for dim in (8, 16):
        obj = PCA(dim, svd_solver="full").fit(train)
        pcas[dim] = obj
        score("pca", lambda w, obj=obj: obj.inverse_transform(obj.transform(w[:, :, 0]))[:, position], dim)
    for width in (3, 5, 9):
        score("causal_median", lambda w, width=width: np.median(w[:, position-width+1:position+1, 0], axis=1), width)
    for width in (5, 9):
        for threshold in (3., 5.):
            score("causal_hampel", lambda w, width=width, threshold=threshold:
                  baseline_predictions(w, mean, pcas[8], 3, (width, threshold),
                                       endpoint_offset=p.correction_delay)["causal_hampel"][:, position],
                  [width, threshold])
    for threshold in (3., 5., 7., 9.):
        score("global_mad", lambda w, threshold=threshold: np.where(
            np.abs(w[:, position, 0]-global_center) > threshold*global_scale, global_center, w[:, position, 0]), threshold)
    score("observed_input", lambda w: w[:, position, 0], None)
    score("train_mean", lambda w: np.full(len(w), mean), None)
    names = ["pca", "causal_median", "causal_hampel", "global_mad", "observed_input", "train_mean"]
    if p.correction_delay:
        score("centered_median", lambda w: np.median(w[:, position-1:position+2, 0], axis=1), 3)
        for threshold in (3., 5., 7., 9.):
            score("delayed_pair", lambda w, threshold=threshold: delayed_pair_values(w[:, :, 0], threshold, [position])[:, position], threshold)
        names.extend(["centered_median", "delayed_pair"])
    choices = {}
    for name in names:
        candidates = [r for r in scores if r["model"] == name]
        feasible = [r for r in candidates if r["normal_mse_ratio"] <= p.guard_limit]
        choices[name] = min(feasible or candidates, key=lambda r: r["mae_ratio"])
    feasible = [r for r in choices.values() if r["normal_mse_ratio"] <= p.guard_limit]
    reference = min(feasible, key=lambda r: r["mae_ratio"])["model"]
    selected_pca = pcas[choices["pca"]["setting"]]
    save_npz(out/"pca.npz", dict(mean=selected_pca.mean_, components=selected_pca.components_))
    result = dict(candidates=scores, choices=choices, reference_baseline=reference, mean=mean,
                  selection="lowest validation MAE among guard-feasible settings, if any; else lowest MAE",
                  global_center=global_center, global_scale=global_scale)
    write_json(out/"baseline_selection.json", result)
    return selected_pca, result


def global_mad(values, center, scale, threshold):
    return np.where(np.abs(values-center) > threshold*scale, center, values)


def representation_baseline_features(windows, clean_windows, fitted):
    flat = windows[:, :, 0]
    filtered = global_mad(flat, fitted["center"], fitted["scale"], fitted["threshold"])
    return dict(raw=flat, pca=fitted["pca"].transform(flat),
                mad_pca=fitted["mad_pca"].transform(filtered), clean_oracle=clean_windows[:, :, 0])


def fit_representation_baselines(development, p, out):
    data = {s: pooled_representation_data(development, s, p) for s in ("train", "val")}
    windows = {s: data[s]["windows"] for s in data}
    targets = {s: data[s]["target"] for s in data}
    clean = {s: data[s]["clean"] for s in data}
    train_flat = np.concatenate([window_inputs(t["train"], p, p.training_stride)[:, :, 0] for t in development])
    values = np.concatenate([t["train"]["observed"] for t in development])
    center = float(np.median(values))
    scale = max(float(1.4826*np.median(np.abs(values-center))), 1e-6)
    pca = PCA(p.latent_dim, svd_solver="full").fit(train_flat)
    probes = {}
    for name, x, v in (("raw", windows["train"][:, :, 0], windows["val"][:, :, 0]),
                       ("pca", pca.transform(windows["train"][:, :, 0]), pca.transform(windows["val"][:, :, 0])),
                       ("clean_oracle", clean["train"][:, :, 0], clean["val"][:, :, 0])):
        probes[name] = fit_probe(x, targets["train"], v, targets["val"])
    candidates, best = [], None
    for threshold in p.mad_thresholds:
        candidate_pca = PCA(p.latent_dim, svd_solver="full").fit(global_mad(train_flat, center, scale, threshold))
        x = candidate_pca.transform(global_mad(windows["train"][:, :, 0], center, scale, threshold))
        v = candidate_pca.transform(global_mad(windows["val"][:, :, 0], center, scale, threshold))
        probe = fit_probe(x, targets["train"], v, targets["val"])
        candidates.append(dict(threshold=threshold, probe=probe))
        if best is None or probe["validation_mae"] < best[0]:
            best = probe["validation_mae"], threshold, candidate_pca, probe
    _, threshold, mad_pca, probes["mad_pca"] = best
    receipt = dict(center=center, scale=scale, threshold=threshold, probes=probes,
                   candidates=candidates, mean=float(targets["train"].mean()),
                   median=float(np.median(targets["train"])))
    save_npz(out/"representation_pca.npz", dict(pca_mean=pca.mean_, pca_components=pca.components_,
        mad_pca_mean=mad_pca.mean_, mad_pca_components=mad_pca.components_))
    write_json(out/"representation_baselines.json", receipt)
    return dict(**receipt, pca=pca, mad_pca=mad_pca)

