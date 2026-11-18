"""Ошибки исправления, бутстреп по траекториям и проверка критерия успеха."""
import numpy as np


def endpoint_metrics(prediction, clean, anomaly, *, large_threshold=None):
    prediction, clean, anomaly = np.asarray(prediction), np.asarray(clean), np.asarray(anomaly, bool)
    if prediction.shape != clean.shape or prediction.ndim != 1 or anomaly.shape != clean.shape:
        raise ValueError("expected aligned endpoint vectors")
    if not len(clean) or not np.isfinite(prediction).all() or not np.isfinite(clean).all():
        raise ValueError("empty or nonfinite predictions")
    result = {}
    masks = [("all", np.ones(len(clean), bool)), ("normal", ~anomaly), ("contaminated", anomaly)]
    if large_threshold is not None:
        if not np.isfinite(large_threshold) or large_threshold < 0:
            raise ValueError("invalid large-move threshold")
        masks.extend([("large", np.abs(clean) >= large_threshold),
                      ("large_normal", (np.abs(clean) >= large_threshold) & ~anomaly)])
    for label, mask in masks:
        error = prediction[mask]-clean[mask]
        result[f"{label}_count"] = int(mask.sum())
        result[f"{label}_mae"] = float(np.abs(error).mean()) if len(error) else None
        result[f"{label}_mse"] = float(np.square(error).mean()) if len(error) else None
    return result


def relative_scores(metrics, reference):
    values = []
    for key in ("all_mae", "normal_mse"):
        denominator = reference[key]
        if denominator is None or denominator <= 1e-12 or metrics[key] is None:
            raise ValueError("undefined trajectory ratio; confirmation cannot pass")
        values.append(metrics[key]/denominator)
    return values


def trajectory_interval(ratios, repeats=5000, seed=61017, *, upper_quantile=.975):
    """Парный бутстреп по траекториям: строки задают семейства, столбцы траектории.

    Перед вызовом ошибки инициализаций усредняются внутри каждой траектории.
    Интервалы условны на данных обучения и выбранных моделях.
    """
    ratios = np.asarray(ratios, float)
    if not .5 < upper_quantile < 1:
        raise ValueError("upper quantile must be between .5 and 1")
    if ratios.ndim != 2 or ratios.shape[1] < 2 or not np.isfinite(ratios).all():
        raise ValueError("need at least two independent trajectories per family")
    rng = np.random.default_rng(seed)
    samples = np.zeros(repeats)
    for family in ratios:
        indices = rng.integers(0, len(family), (repeats, len(family)))
        samples += family[indices].mean(axis=1)/len(ratios)
    return dict(mean=float(ratios.mean()), lower=float(np.quantile(samples, 1-upper_quantile)),
                upper=float(np.quantile(samples, upper_quantile)), unit="independent_trajectory",
                families=ratios.shape[0], trajectories_per_family=ratios.shape[1],
                repeats=repeats, conditional_on="development data and fixed initialization seeds")


def validation_guard(ratios, development, p):
    """Worst family/volatility validation risk; optional bootstrap selection heuristic."""
    ratios = np.asarray(ratios, float)
    if ratios.shape != (len(development), 2) or not np.isfinite(ratios).all():
        raise ValueError("invalid trajectory validation ratios")
    if p.guard_scope == "average":
        return float(ratios[:, 1].mean())
    bounds = []
    for family in p.families:
        for scale in p.volatility_scales:
            group = [i for i, item in enumerate(development)
                     if item["family"] == family and item.get("volatility_scale", 1.) == scale]
            if not group:
                raise ValueError("missing family/volatility validation stratum")
            values = ratios[group, 1]
            if p.validation_upper_quantile is None:
                bounds.append(float(values.mean()))
            else:
                bounds.append(trajectory_interval(values[None], p.validation_bootstrap_repeats,
                    p.bootstrap_seed, upper_quantile=p.validation_upper_quantile)["upper"])
    return max(bounds)


def confirmation_decision(metrics, p, baseline):
    table = metrics.groupby(["family", "data_seed", "model"])[["all_mae", "normal_mse"]].mean()
    result = {}
    quantile = 1-.05/(1+len(p.families)) if p.guard_scope == "family" else .975
    for method in metrics.model.unique():
        comparisons = {}
        for reference in dict.fromkeys(("observed_input", baseline)):
            intervals = {}
            for key in ("all_mae", "normal_mse"):
                ratios = []
                for family in p.families:
                    ratios.append([table.loc[(family, seed, method), key]/table.loc[(family, seed, reference), key]
                                   for seed in p.confirmation_seeds])
                intervals[key] = trajectory_interval(ratios, p.bootstrap_repeats, p.bootstrap_seed,
                                                     upper_quantile=quantile)
            comparisons[reference] = intervals
        primary = comparisons["observed_input"]
        result[method] = dict(comparisons=comparisons,
            joint_success=bool(not p.smoke and primary["all_mae"]["upper"] < 1 and
                               primary["normal_mse"]["upper"] <= p.guard_limit))
        if p.guard_scope == "family":
            guards = {}
            for family in p.families:
                ratios = [[table.loc[(family, seed, method), "normal_mse"] /
                           table.loc[(family, seed, "observed_input"), "normal_mse"] for seed in p.confirmation_seeds]]
                guards[family] = trajectory_interval(ratios, p.bootstrap_repeats, p.bootstrap_seed,
                                                     upper_quantile=quantile)
            result[method]["family_normal_mse"] = guards
            result[method]["joint_success"] = bool(not p.smoke and primary["all_mae"]["upper"] < 1 and
                                                    all(v["upper"] <= p.guard_limit for v in guards.values()))
    return dict(methods=result, reference_baseline=baseline, smoke=p.smoke,
                decision=(f"MAE/input upper bound < 1; normal MSE/input upper bound <= {p.guard_limit} in every family"
                          if p.guard_scope == "family" else
                          f"both one-sided 97.5% upper bounds: MAE ratio < 1 and normal MSE ratio <= {p.guard_limit}"),
                multiplicity=(f"Bonferroni across {1+len(p.families)} primary criteria; baseline comparisons descriptive"
                              if p.guard_scope == "family" else
                              "Bonferroni across the two primary criteria; baseline comparisons descriptive"),
                model_repeat_policy="average metrics over initialization seeds; predictions are not ensembled")


def representation_decision(frame, p):
    """Three predeclared paired comparisons at the standard contamination level."""
    methods = [(m, -1) for m in ("raw", "pca", "mad_pca", "clean_oracle", "mean", "median")]
    methods += [(m, seed) for m in ("ordinary", "denoising") for seed in p.initialization_seeds]
    expected = {(family, seed, severity, method, init) for family in p.families
                for seed in p.confirmation_seeds for severity in (0, 1, 2) for method, init in methods}
    keys = ["family", "data_seed", "severity", "model", "initialization_seed"]
    if frame.duplicated(keys).any() or set(frame[keys].itertuples(index=False, name=None)) != expected:
        raise ValueError("incomplete or duplicate representation confirmation grid")
    if not np.isfinite(frame[["mae", "mse", "r2"]].to_numpy()).all() or not (frame["count"] > 0).all():
        raise ValueError("invalid forecast results")
    table = frame[frame.severity == 1].groupby(["family", "data_seed", "model"])["mae"].mean()
    comparisons = {}
    quantile = 1-.05/3
    for reference in ("ordinary", "pca", "mad_pca"):
        ratios = []
        for family in p.families:
            family_ratios = []
            for seed in p.confirmation_seeds:
                denominator = table.loc[(family, seed, reference)]
                if denominator <= 1e-12:
                    raise ValueError("undefined primary forecast ratio")
                
                family_ratios.append(table.loc[(family, seed, "denoising")]/denominator)
            ratios.append(family_ratios)
        comparisons[reference] = trajectory_interval(ratios, p.bootstrap_repeats, p.bootstrap_seed,
                                                     upper_quantile=quantile)
    return dict(primary="denoising AE forecast MAE", severity=1, comparisons=comparisons,
                upper_quantile=quantile, multiplicity="Bonferroni across three comparisons, family alpha .05",
                joint_success=bool(not p.smoke and all(r["upper"] < 1 for r in comparisons.values())),
                smoke=p.smoke, seed_policy="mean metrics within trajectory; no prediction ensemble",
                stress_policy="descriptive paired conditions 0/1/2, no retuning")

