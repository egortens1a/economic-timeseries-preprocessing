"""Сравнение направления TCN, календарных признаков и бустинга при заполнении пропусков.

python scripts/check_nn5_components.py --data dataset.zip --out results/nn5_components
"""
import argparse
from dataclasses import asdict, replace
import hashlib
from pathlib import Path
import pickle
import shutil
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from tsprep.config import ImputationProtocol
from tsprep.imputation import data as d, baselines as f, training as n
from tsprep.imputation.evaluation import bootstrap_indices, bootstrap_ratio
from tsprep.imputation_experiment import fit_baselines, metric
from tsprep.mlcore.trainer import seed_everything
from tsprep.models.imputer import MaskedTCN
from tsprep.provenance import (file_hash, runtime_environment, save_npz,
                               source_hashes, source_snapshot, write_json)


ARMS = ("full", "forward", "no_phase")
BOOSTING_RATES = (.03, .1)


class _ForwardTCN(MaskedTCN):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Conv1d(16, 1, 1)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)

    def forward(self, x):
        return self.head(self.backbone(x))[:, 0]


def without_phase(data, scaler, base):
    inputs = f.network_features(data, scaler, base)
    inputs[..., 3:] = 0.
    return inputs


def build_arm(arm, seed, device):
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    seed_everything(seed, deterministic=True)
    model = (_ForwardTCN() if arm == "forward" else MaskedTCN()).to(device)
    return model, without_phase if arm == "no_phase" else f.network_features


def backbone_hash(model):
    digest = hashlib.sha256()
    for key, value in model.backbone.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def fit_arms(train, val, scaler, fitted, best, base, out, p):
    selected, models, training = {}, {}, []
    for arm in ARMS:
        candidates, by_rate = [], {}
        for lr in p.learning_rates:
            for seed in p.initialization_seeds:
                model, features = build_arm(arm, seed, p.device)
                initial_hash = backbone_hash(model)
                model, receipt = n._fit_model(model, features, train, val, scaler, fitted,
                                              best, base, lr, seed, p.device, p)
                folder = out/"models"/arm/f"lr{lr}_seed{seed}"
                folder.mkdir(parents=True)
                torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, folder/"model.pt")
                receipt.update(arm=arm, initial_backbone_sha256=initial_hash)
                write_json(folder/"training.json", receipt)
                training.append({k: v for k, v in receipt.items() if k != "history"})
                by_rate[lr, seed] = model
                print(f"FIT {arm} lr={lr} seed={seed} epoch={receipt['best_epoch']} val={receipt['validation_score']:.6f}", flush=True)
            for shrink in p.shrinkages:
                scores = [d.error_blocks(n._predict(by_rate[lr, seed], features, val, scaler,
                            base, p.device, shrink), val, scaler)["score"] for seed in p.initialization_seeds]
                candidates.append(dict(learning_rate=lr, shrinkage=shrink,
                                       score=float(np.mean(scores)), seed_scores=scores))
        chosen = min(candidates, key=lambda item: item["score"])
        selected[arm] = chosen
        models[arm] = {seed: by_rate[chosen["learning_rate"], seed] for seed in p.initialization_seeds}
        write_json(out/f"{arm}_selection.json", dict(selected=chosen, candidates=candidates))
    for seed in p.initialization_seeds:
        assert len({r["initial_backbone_sha256"] for r in training if r["seed"] == seed}) == 1
    pd.DataFrame(training).to_csv(out/"training_summary.csv", index=False)
    return models, selected


def boosting_fill(model, data, scaler, shrink=1.):
    value = f.simple("linear", data, scaler)
    rows, columns = np.nonzero(~np.isfinite(data["windows"]))
    if len(rows) and shrink:
        x, base = f.gap_features(data, scaler, rows, columns)
        value[rows, columns] = base+shrink*model.predict(x)
    return d.clip_missing(value, data["windows"], data["atm"], scaler)


def fit_boosting(train, val, scaler, out, p):
    features, targets = [], []
    for seed in (21001, 21002, 21003):
        extra = d.sample_mask(train["windows"], np.random.default_rng(seed), scaler, p.missing_fraction)
        masked = train["windows"].copy()
        masked[extra] = np.nan
        rows, columns = np.nonzero(extra)
        x, base = f.gap_features(dict(train, windows=masked), scaler, rows, columns)
        features.append(x)
        targets.append(train["windows"][rows, columns]-base)
    x, y = np.concatenate(features), np.concatenate(targets)
    candidates, models = [], {}
    for lr in BOOSTING_RATES:
        model = HistGradientBoostingRegressor(learning_rate=lr, max_iter=150,
                    max_leaf_nodes=15, min_samples_leaf=20, early_stopping=False,
                    loss="squared_error", random_state=21001).fit(x, y)
        models[lr] = model
        for shrink in p.shrinkages:
            score = d.error_blocks(boosting_fill(model, val, scaler, shrink), val, scaler)["score"]
            candidates.append(dict(learning_rate=lr, shrinkage=shrink, score=score))
    selected = min(candidates, key=lambda item: item["score"])
    model = models[selected["learning_rate"]]
    with (out/"models/boosting.pkl").open("wb") as stream:
        pickle.dump(model, stream)
    write_json(out/"boosting_selection.json", dict(selected=selected, candidates=candidates,
               parameters=model.get_params(), training_rows=len(y), features=x.shape[1],
               training_mask_seeds=[21001, 21002, 21003]))
    print("BOOSTING", selected, flush=True)
    return model, selected


def compare(blocks, p):
    full = blocks["full"]["scaled"].sum(axis=0)
    atm, weeks = bootstrap_indices(*full.shape, p.bootstrap_repeats, p.bootstrap_seed, p.bootstrap_block)
    comparisons = {}
    for name in ("forward", "no_phase", "boosting"):
        reference = blocks[name]["scaled"].sum(axis=0)
        ratio = float(full.sum()/reference.sum()) if reference.sum() > 0 else None
        try:
            samples = bootstrap_ratio(full, reference, atm, weeks)
        except ValueError as exc:
            comparisons[name] = dict(ratio=ratio, status="undefined", reason=str(exc))
            continue
        comparisons[name] = dict(ratio=ratio, improvement_pct=100*(1-ratio),
            nominal_two_sided_95pct=np.quantile(samples, [.025, .975]).tolist(),
            corrected_one_sided_upper=float(np.quantile(samples, 1-.05/3)))
    return dict(comparisons=comparisons, scope="exploratory comparisons on previously examined NN5 test",
                uncertainty="paired series and common circular time blocks; fixed fits and masks"), dict(atm=atm, weeks=weeks)


def run(data_path, out, p=ImputationProtocol()):
    data_path, out = Path(data_path), Path(out)
    panel = d.load_panel(data_path)
    raw = panel["values"]
    p.validate(raw.shape[1])
    scaler, normalized = d.fit_scaler(raw, p)
    out.mkdir(parents=True, exist_ok=False)
    begin = time.perf_counter()
    with threadpool_limits(limits=p.threads):
        torch.set_num_threads(p.threads)
        seed_everything(p.initialization_seeds[0], deterministic=True)
        for folder in ("inputs", "models", "validation", "test"):
            (out/folder).mkdir()
        shutil.copy2(data_path, out/"inputs"/("original"+data_path.suffix.lower()))
        save_npz(out/"inputs/panel.npz", panel)
        save_npz(out/"scaler.npz", scaler)
        script_hash, hashes = file_hash(__file__), source_hashes()
        protocol = dict(**asdict(p), length=raw.shape[1], series=raw.shape[0], arms=ARMS,
            protocol_id="imputation_components_v1", input_sha256=file_hash(data_path),
            package_sources=hashes, script_sha256=script_hash,
            direction="forward neural pass only; seasonal baseline still uses the entire available window",
            calendar="zero sine/cosine channels; preserve baseline, profile and five-channel architecture",
            selection="checkpoints, rates and shrinkage selected on validation before test masks are made",
            boosting_rates=BOOSTING_RATES, boosting_training_masks=[21001, 21002, 21003],
            inference="average errors across initializations; no prediction ensemble",
            scope="exploratory component analysis; test period was examined in earlier experiments")
        write_json(out/"protocol.json", protocol)
        source_snapshot(out/"source.zip")
        write_json(out/"environment.json", runtime_environment(p.device))
        print("PROTOCOL FROZEN", file_hash(out/"protocol.json"), flush=True)
        train = d.train_windows(normalized, p)
        train["truth_raw"] = np.stack([raw[s, a:a+p.window] for s, a in zip(train["atm"], train["start"])])
        val_masks = d.additional_masks(normalized, scaler, p.train_stop, p.val_stop,
                                       p.validation_mask_seeds, p.missing_fraction)
        validation = d.evaluation_windows(raw, normalized, val_masks, p.train_stop, p.val_stop, p)
        save_npz(out/"validation/masks.npz", dict(extra=val_masks))
        save_npz(out/"validation/windows.npz", validation)
        fitted, val_predictions, best = fit_baselines(train, validation, scaler, out, p)
        models, selected = fit_arms(train, validation, scaler, fitted, best, val_predictions[best], out, p)
        booster, boosting = fit_boosting(train, validation, scaler, out, p)
        for arm in ARMS:
            features = without_phase if arm == "no_phase" else f.network_features
            for seed, model in models[arm].items():
                val_predictions[f"{arm}_{seed}"] = n._predict(model, features, validation, scaler,
                    val_predictions[best], p.device, selected[arm]["shrinkage"])
        val_predictions["boosting"] = boosting_fill(booster, validation, scaler, boosting["shrinkage"])
        save_npz(out/"validation/predictions.npz", val_predictions)
        frozen = {path.relative_to(out).as_posix(): file_hash(path)
                  for path in sorted(out.rglob("*")) if path.is_file()}
        write_json(out/"selection.json", dict(baseline=best, neural=selected, boosting=boosting, files=frozen))
        selection_hash = file_hash(out/"selection.json")
        print("SELECTION FROZEN", selected, flush=True)
        masks = d.additional_masks(normalized, scaler, p.val_stop, raw.shape[1],
                                   p.test_mask_seeds, p.missing_fraction)
        test = d.evaluation_windows(raw, normalized, masks, p.val_stop, raw.shape[1], p)
        save_npz(out/"test/masks.npz", dict(extra=masks))
        save_npz(out/"test/windows.npz", test)
        base = f.predict(best, test, scaler, fitted)
        predictions = {name: f.predict(name, test, scaler, fitted) for name in f.METHODS}
        predictions["boosting"] = boosting_fill(booster, test, scaler, boosting["shrinkage"])
        for arm in ARMS:
            features = without_phase if arm == "no_phase" else f.network_features
            for seed, model in models[arm].items():
                predictions[f"{arm}_{seed}"] = n._predict(model, features, test, scaler,
                    base, p.device, selected[arm]["shrinkage"])
        save_npz(out/"test/predictions.npz", predictions)
        blocks = {name: d.error_blocks(value, test, scaler) for name, value in predictions.items()}
        for arm in ARMS:
            blocks[arm] = {key: np.mean([blocks[f"{arm}_{seed}"][key] for seed in p.initialization_seeds], axis=0)
                           for key in ("scaled", "absolute", "squared", "count")}
        pd.DataFrame([metric(name, value) for name, value in blocks.items()]).to_csv(out/"summary.csv", index=False)
        save_npz(out/"test/error_blocks.npz", {name+"__"+key: value[key] for name, value in blocks.items()
                                              for key in ("scaled", "absolute", "squared", "count")})
        decision, bootstrap = compare(blocks, p)
        write_json(out/"comparisons.json", decision)
        save_npz(out/"bootstrap_indices.npz", bootstrap)
        if source_hashes() != hashes or file_hash(__file__) != script_hash or file_hash(out/"selection.json") != selection_hash:
            raise RuntimeError("source or selection changed during the experiment")
        for name, digest in frozen.items():
            if file_hash(out/name) != digest:
                raise RuntimeError(f"frozen input changed: {name}")
        files = {path.relative_to(out).as_posix(): file_hash(path) for path in sorted(out.rglob("*")) if path.is_file()}
        write_json(out/"complete.json", dict(status="complete", seconds=time.perf_counter()-begin,
                   selection_sha256=selection_hash, files=files))
        print("COMPARISONS", decision, flush=True)
        print("SAVED", out, flush=True)
        return decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Исходный ZIP NN5 или NPZ с values[series,time]")
    parser.add_argument("--out", required=True, help="Новая папка результатов")
    defaults = ImputationProtocol()
    for name in ("window", "period", "train_stop", "val_stop", "epochs", "batch_size", "bootstrap_repeats", "threads"):
        parser.add_argument("--"+name.replace("_", "-"), type=int, default=getattr(defaults, name))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke", action="store_true", help="Одна эпоха и одна инициализация")
    args = vars(parser.parse_args())
    data, out, smoke = args.pop("data"), args.pop("out"), args.pop("smoke")
    p = ImputationProtocol(**args)
    if smoke:
        p = replace(p, epochs=1, initialization_seeds=p.initialization_seeds[:1],
                    learning_rates=p.learning_rates[:1], bootstrap_repeats=100)
    run(data, out, p)


if __name__ == "__main__":
    main()
