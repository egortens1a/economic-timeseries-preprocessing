"""Обучение моделей исправления и выбор весов на проверочной выборке."""
import io
import time
from dataclasses import asdict
import numpy as np
import torch
from tsprep.data_preprocessing.datasets import sliding_windows
from tsprep.data_preprocessing.denoising import FAMILIES, generate_denoising_series
from tsprep.data_preprocessing.preprocessing import TrainFittedNormalizer
from tsprep.denoising.data import window_inputs, denoising_indices
from tsprep.denoising.metrics import endpoint_metrics, relative_scores, validation_guard
from tsprep.mlcore.trainer import seed_everything
from tsprep.models.denoiser import ResidualTCNDenoiser
from tsprep.mlcore.losses import residual_loss
from tsprep.provenance import atomic_write, file_hash, write_json
from tsprep.config import TrainConfig
from tsprep.data_preprocessing.datasets import TensorWindowLoader
from tsprep.denoising.data import pooled_representation_data
from tsprep.evaluation.downstream import fit_probe, predict_probe
from tsprep.evaluation.reconstruction import collect_embeddings
from tsprep.mlcore.trainer import train_model
from tsprep.provenance import input_fingerprint, save_npz


@torch.no_grad()
def predict(model, windows, device, batch_size=512):
    model.eval()
    output = []
    for start in range(0, len(windows), batch_size):
        x = torch.as_tensor(windows[start:start+batch_size], device=device)
        output.append(model(x).cpu().numpy())
    return np.concatenate(output)


def build_denoiser(p, seed):
    seed_everything(seed, deterministic=True)
    return ResidualTCNDenoiser(hidden_dim=p.hidden_dim, levels=p.levels,
                              local_context=p.local_context, context_mode=p.context_mode,
                              correction_delay=p.correction_delay, delay_mode=p.delay_mode).to(p.device)


def save_checkpoint(path, model):
    buffer = io.BytesIO()
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, buffer)
    atomic_write(path, buffer.getvalue())


def overfit_check(p, out):
    """Debug seed only; optimize 16 normal and 16 contaminated windows."""
    series = generate_denoising_series(FAMILIES[0], 9002, max(p.length, 2400))
    x = np.diff(np.log(series.values), axis=0)
    y = np.diff(np.log(series.clean_values), axis=0)
    normalizer = TrainFittedNormalizer().fit(x)
    windows = np.array(sliding_windows(normalizer.transform(x), p.window), np.float32)
    ends = np.arange(p.window-1, len(x))-p.correction_delay
    targets = normalizer.transform(y)[ends, 0].astype(np.float32)
    mask = (series.is_anomaly[1:] | series.is_anomaly[:-1])[ends]
    rng = np.random.default_rng(9002)
    indices = np.concatenate([rng.choice(np.flatnonzero(mask == kind), 16, replace=False)
                              for kind in (False, True)])
    model = build_denoiser(p, 111)
    x_t = torch.as_tensor(windows[indices], device=p.device)
    y_t = torch.as_tensor(targets[indices], device=p.device)
    opt = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=0.)
    initial = float(torch.mean((model.reference(x_t)[:, -1, 0]-y_t)**2).item())
    steps = 2 if p.smoke else 800
    losses = []
    for _ in range(steps):
        model.train()
        prediction = model(x_t)[:, -1, 0]
        loss = torch.mean((prediction-y_t)**2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        losses.append(float(loss.item()))
    final = float(torch.mean((model(x_t)[:, -1, 0]-y_t)**2).item())
    result = dict(initial_mse=initial, final_mse=final, reduction=1-final/initial,
                  windows=32, contaminated_windows=16, steps=steps, debug_seed=9002,
                  passed=final < .05*initial, smoke=p.smoke, loss=losses)
    write_json(out/"overfit_check.json", result)
    print(f"overfit: MSE {initial:.6f} -> {final:.6f}, passed={result['passed']}", flush=True)
    if not result["passed"] and not p.smoke:
        raise RuntimeError("Small-sample overfit failed; no confirmation data generated")
    return result


def validation_scores(model, validation, p, *, return_details=False):
    ratios = []
    for item in validation:
        segment = item["val"]
        windows = window_inputs(segment, p)
        ends, _ = denoising_indices(segment, p)
        target = segment["clean"][ends, 0]
        mask = segment["anomaly"][ends]
        reference = endpoint_metrics(windows[:, -1-p.correction_delay, 0], target, mask)
        pred = predict(model, windows, p.device, p.batch_size)[:, -1, 0]
        ratios.append(relative_scores(endpoint_metrics(pred, target, mask), reference))
    result = np.mean(ratios, axis=0).tolist()
    result[1] = validation_guard(ratios, validation, p)
    if return_details:
        details = [dict(family=t["family"], data_seed=t.get("validation_seed", t["seed"]),
                        volatility_scale=t.get("volatility_scale", 1.),
                        mae_ratio=r[0], normal_mse_ratio=r[1]) for t, r in zip(validation, ratios)]
        return *result, details
    return result


def fit_candidate(p, development, loss_kind, lr, seed, out, *, identity_weight=0.):
    out.mkdir()
    x = np.concatenate([window_inputs(t["train"], p, p.training_stride) for t in development])
    y = np.concatenate([t["train"]["clean"][denoising_indices(t["train"], p, p.training_stride)[0], 0] for t in development])
    x, y = torch.as_tensor(x, device=p.device), torch.as_tensor(y, device=p.device)
    clean = None
    if loss_kind == "mse_identity" and identity_weight:
        clean = torch.as_tensor(np.concatenate([window_inputs(t["train"], p, p.training_stride, clean_input=True)
                                               for t in development]), device=p.device)
    model = build_denoiser(p, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.0001)
    generator = torch.Generator(device=p.device).manual_seed(seed+10000)
    # На эпохе 0 модель копирует вход. Это запасной вариант, если обучение не дало улучшения.
    best = dict(epoch=0, mae_ratio=1., normal_mse_ratio=1.)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    history = []
    begin = time.perf_counter()
    for epoch in range(1, p.epochs+1):
        order = torch.randperm(len(x), generator=generator, device=p.device)
        total = 0.
        model.train()
        for start in range(0, len(order), p.batch_size):
            indices = order[start:start+p.batch_size]
            prediction = model(x[indices])[:, -1, 0]
            clean_prediction = model(clean[indices])[:, -1, 0] if clean is not None else None
            loss = residual_loss(prediction, y[indices], loss_kind,
                                 clean_prediction=clean_prediction, identity_weight=identity_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            total += loss.item()*len(indices)
        
        if epoch == 1 or epoch % p.validation_every == 0 or epoch == p.epochs:
            mae_ratio, guard_ratio, details = validation_scores(model, development, p, return_details=True)
            row = dict(epoch=epoch, training_loss=total/len(x), mae_ratio=mae_ratio,
                       normal_mse_ratio=guard_ratio)
            if p.validation_seeds or p.volatility_scales != (1.,) or p.validation_upper_quantile is not None:
                row["validation_trajectories"] = details
            history.append(row)
            if guard_ratio <= p.guard_limit and mae_ratio < best["mae_ratio"]:
                best = {k: v for k, v in row.items() if k != "training_loss"}
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            write_json(out/"progress.json", dict(history=history, best=best))
            print(f"{out.name} epoch={epoch}/{p.epochs} valMAE/input={mae_ratio:.4f} "
                  f"normalMSE/input={guard_ratio:.4f} selected={best['epoch']}", flush=True)
    
    model.load_state_dict(best_state)
    save_checkpoint(out/"model.pt", model)
    result = dict(loss=loss_kind, lr=lr, seed=seed, best=best, history=history,
                  seconds=time.perf_counter()-begin, parameters=sum(t.numel() for t in model.parameters()),
                  model_sha256=file_hash(out/"model.pt"))
    
    if loss_kind == "mse_identity":
        result["identity_weight"] = identity_weight
    if p.guard_scope != "average":
        result["guard_scope"] = p.guard_scope
    if p.validation_upper_quantile is not None:
        result["validation_upper_quantile"] = p.validation_upper_quantile
    write_json(out/"training.json", result)
    return model, result


def save_ordinary_control(p, development, out):
    # При обучении observed → observed копирование входа уже даёт нулевую ошибку.
    ordinary = build_denoiser(p, 111)
    check_x = torch.as_tensor(window_inputs(development[0]["train"], p)[:32], device=p.device)
    ordinary_loss = (ordinary(check_x)-ordinary.reference(check_x)).square().mean()
    ordinary_loss.backward()
    if ordinary_loss.item() != 0 or any(t.grad is not None and t.grad.count_nonzero().item() for t in ordinary.parameters()):
        raise AssertionError("ordinary reconstruction must start at exact zero loss/gradient")
    
    save_checkpoint(out/"ordinary_model.pt", ordinary)
    write_json(out/"ordinary_arm.json", dict(loss=0., gradient=0., trained=False,
        reason="Identity is an exact global optimum of observed-to-observed residual reconstruction."))


def select_models(p, development, out):
    candidates = []
    trained = {}
    for loss in p.losses:
        for lr in p.learning_rates:
            for weight in p.identity_weights if loss == "mse_identity" else (0.,):
                suffix = f"_lambda{weight:g}" if loss == "mse_identity" else ""
                name = f"candidate_{loss}{suffix}_lr{lr:g}_seed111"
                model, receipt = fit_candidate(p, development, loss, lr, 111, out/name, identity_weight=weight)
                candidates.append(dict(path=name, **receipt))
                trained[name] = model
    
    selected = min(candidates, key=lambda r: (r["best"]["mae_ratio"], r["lr"], r["loss"], r.get("identity_weight", 0.)))
    models = {111: trained[selected["path"]]}
    receipts = {"111": selected}
    for seed in p.initialization_seeds:
        if seed == 111: continue
        name = f"selected_seed{seed}"
        models[seed], receipts[str(seed)] = fit_candidate(p, development, selected["loss"], selected["lr"], seed, out/name,
                                                        identity_weight=selected.get("identity_weight", 0.))
    return models, dict(selected=selected, candidates=candidates, final_models=receipts)


def build_embedding_model(p, seed):
    from tsprep.config import ModelConfig
    from tsprep.models.autoencoder import build_model
    seed_everything(seed, deterministic=True)
    config = ModelConfig(model_name="tcn_ae", latent_dim=p.latent_dim, hidden_dim=p.hidden_dim,
                         tcn_levels=p.levels, dropout=0, upsampling="matmul")
    return build_model(config, 1, p.window).to(p.device)


def check_embedding_learning(p, out):
    from tsprep.config import TrainConfig
    from tsprep.data_preprocessing.datasets import TensorWindowLoader
    from tsprep.mlcore.trainer import train_model
    series = generate_denoising_series(FAMILIES[0], 9002, max(p.length, 2400))
    x = np.diff(np.log(series.values), axis=0)
    normalizer = TrainFittedNormalizer().fit(x)
    x = np.array(sliding_windows(normalizer.transform(x), p.window), np.float32)
    clean = np.array(sliding_windows(normalizer.transform(np.diff(np.log(series.clean_values), axis=0)), p.window), np.float32)
    mask = (series.is_anomaly[1:] | series.is_anomaly[:-1])[p.window-1:]
    rng = np.random.default_rng(9002)
    indices = np.concatenate([rng.choice(np.flatnonzero(mask == k), 16, replace=False) for k in (False, True)])
    results = {}
    for arm, target in (("ordinary", x), ("denoising", clean)):
        model = build_embedding_model(p, 111)
        loader = TensorWindowLoader(x[indices], 32, p.device, targets=target[indices])
        cfg = TrainConfig(epochs=2 if p.smoke else 400, lr=.003, weight_decay=0.,
                          early_stopping_patience=400, deterministic=True, device=p.device, seed=111,
                          training_target="observed" if arm == "ordinary" else "clean",
                          validation_target="observed" if arm == "ordinary" else "clean")
        history = train_model(model, loader, loader, cfg, verbose=False)
        passed = bool(history.best_score < history.train_loss[0])
        results[arm] = dict(initial_mse=history.train_loss[0], final_mse=history.best_score,
                            steps=len(history.train_loss), passed=passed)
    write_json(out/"embedding_sanity.json", results)
    if not p.smoke and not all(r["passed"] for r in results.values()):
        raise RuntimeError("Embedding learning sanity failed before confirmation")
    print("Embedding sanity:", results, flush=True)


def fit_embedding(p, development, baseline, arm, lr, seed, out):
    out.mkdir()
    model = build_embedding_model(p, seed)
    initial_hash = input_fingerprint({k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}, ["weights"])
    x = {s: np.concatenate([window_inputs(t[s], p, p.training_stride if s == "train" else 1)
                            for t in development]) for s in ("train", "val")}
    clean = {s: np.concatenate([window_inputs(t[s], p, p.training_stride if s == "train" else 1, clean_input=True)
                                for t in development]) for s in ("train", "val")}
    loaders = {s: TensorWindowLoader(x[s], p.batch_size, p.device,
        targets=x[s] if s == "train" and arm == "ordinary" else clean[s],
        shuffle=s == "train", seed=seed+10000) for s in x}
    cfg = TrainConfig(epochs=p.epochs, lr=lr, early_stopping_patience=p.epochs,
        device=p.device, seed=seed, deterministic=True,
        training_target="observed" if arm == "ordinary" else "clean", validation_target="clean")
    print(f"FIT {out.name}: {p.epochs} epochs", flush=True)
    begin = time.perf_counter()
    history = train_model(model, loaders["train"], loaders["val"], cfg, verbose=not p.smoke)
    data = {s: pooled_representation_data(development, s, p) for s in ("train", "val")}
    z = {s: collect_embeddings(model, TensorWindowLoader(data[s]["windows"], p.batch_size, p.device), p.device)["z"] for s in data}
    probe = fit_probe(z["train"], data["train"]["target"], z["val"], data["val"]["target"])
    prediction = predict_probe(probe, z["val"])
    raw = predict_probe(baseline["probes"]["raw"], data["val"]["windows"][:, :, 0])
    validation = []
    for group in data["val"]["groups"]:
        sl = slice(group["start"], group["stop"])
        y = data["val"]["target"][sl]
        denominator = float(np.abs(raw[sl]-y).mean())
        if denominator <= 1e-12:
            raise ValueError("undefined raw validation forecast MAE")
        
        validation.append(dict(**group, mae=float(np.abs(prediction[sl]-y).mean()), raw_mae=denominator))
    
    save_checkpoint(out/"model.pt", model)
    save_npz(out/"embeddings.npz", dict(train_z=z["train"], val_z=z["val"],
        train_target=data["train"]["target"], val_target=data["val"]["target"],
        train_endpoint=data["train"]["ends"], val_endpoint=data["val"]["ends"]))
    
    result = dict(path=out.name, arm=arm, lr=lr, seed=seed, loader_seed=seed+10000,
        initial_state_sha256=initial_hash, model_sha256=file_hash(out/"model.pt"),
        config=asdict(cfg), history=asdict(history), probe=probe, validation=validation,
        validation_forecast_ratio=float(np.mean([r["mae"]/r["raw_mae"] for r in validation])),
        train_groups=data["train"]["groups"], seconds=time.perf_counter()-begin,
        parameters=sum(v.numel() for v in model.parameters()))
    
    write_json(out/"training.json", result)
    print(f"FITTED {out.name}: val forecast/raw={result['validation_forecast_ratio']:.5f}", flush=True)
    return model, result


def select_embedding_models(p, development, baseline, out):
    models, receipts, candidates = {}, {}, []
    # Выбираем скорость обучения на инициализации 111.
    for lr in p.learning_rates:
        for arm in ("ordinary", "denoising"):
            key = f"{arm}_lr{lr:g}_seed111"
            models[key], receipts[key] = fit_embedding(p, development, baseline, arm, lr, 111, out/key)
            candidates.append(receipts[key])

    lr = min(p.learning_rates, key=lambda rate: (
        np.mean([r["validation_forecast_ratio"] for r in candidates if r["lr"] == rate]), rate))
    # Модели 111 уже обучены выше; остальные инициализации обучаем с выбранной скоростью.
    chosen, final = {}, {}
    for seed in p.initialization_seeds:
        for arm in ("ordinary", "denoising"):
            name = f"{arm}_seed{seed}"
            if seed == 111:
                key = f"{arm}_lr{lr:g}_seed111"
                chosen[name], final[name] = models[key], receipts[key]
            else:
                chosen[name], final[name] = fit_embedding(p, development, baseline, arm, lr, seed, out/name)
        
        assert final[f"ordinary_seed{seed}"]["initial_state_sha256"] == final[f"denoising_seed{seed}"]["initial_state_sha256"]
    
    return chosen, dict(learning_rate=lr, candidates=candidates, final_models=final)
