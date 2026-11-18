"""Обучение TCN, которая добавляет поправку к базовому заполнению пропусков."""
import copy
import time

import numpy as np
import torch

from .data import clip_missing, error_blocks, sample_mask
from .baselines import network_features, predict as baseline_predict
from tsprep.mlcore.trainer import seed_everything
from tsprep.models.imputer import MaskedTCN
from tsprep.config import ImputationProtocol

def build(seed, device):
    seed_everything(seed, deterministic=True)
    return MaskedTCN().to(device)


@torch.no_grad()
def predict(model, data, scaler, base, device, shrink=1., batch=512):
    return _predict(model, network_features, data, scaler, base, device, shrink, batch)


@torch.no_grad()
def _predict(model, features, data, scaler, base, device, shrink=1., batch=512):
    if shrink == 0:
        return base.copy()
   
    model.eval()
    inputs = features(data, scaler, base)
    corrections = []
    for start in range(0, len(base), batch):
        x = torch.as_tensor(inputs[start:start+batch], device=device)
        corrections.append(model(x).cpu().numpy().astype(float))
    
    value = base+shrink*np.concatenate(corrections)
    return clip_missing(value, data["windows"], data["atm"], scaler)


def fit(train, validation, scaler, fitted, base_method, validation_base, lr, seed, device, p=ImputationProtocol()):
    model = build(seed, device)
    assert sum(p.numel() for p in model.parameters()) == 4305
    return _fit_model(model, network_features, train, validation, scaler, fitted,
                      base_method, validation_base, lr, seed, device, p)


def _fit_model(model, features, train, validation, scaler, fitted, base_method, validation_base,
               lr, seed, device, p=ImputationProtocol()):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.001)
    atm = train["atm"]
    mean = torch.as_tensor(scaler["mean"][atm, None], dtype=torch.float32, device=device)
    std = torch.as_tensor(scaler["std"][atm, None], dtype=torch.float32, device=device)
    scale = torch.as_tensor(scaler["mase"][atm, None], dtype=torch.float32, device=device)
    target = torch.as_tensor(np.nan_to_num(train["truth_raw"]), dtype=torch.float32, device=device)
    order_rng = np.random.default_rng(22001)  # Independent from initialization seeds.
    initial = error_blocks(validation_base, validation, scaler)["score"]
    best_score, best_epoch, best = initial, 0, copy.deepcopy(model.state_dict())
    history = [dict(epoch=0, train_loss=None, validation_score=initial)]
    begin = time.perf_counter()
    for epoch in range(1, p.epochs+1):
        # Маски и порядок окон одинаковы для всех инициализаций.
        hidden = sample_mask(train["windows"], np.random.default_rng(21001+epoch), scaler, p.missing_fraction)
        masked = train["windows"].copy()
        masked[hidden] = np.nan
        case = dict(train, windows=masked)
        base = baseline_predict(base_method, case, scaler, fitted)
        inputs = torch.as_tensor(features(case, scaler, base), device=device)
        base_tensor = torch.as_tensor(base, dtype=torch.float32, device=device)
        mask_tensor = torch.as_tensor(hidden, device=device)
        order = order_rng.permutation(len(base))
        total, count = 0., 0
        model.train()
        for start in range(0, len(order), p.batch_size):
            indices = torch.as_tensor(order[start:start+p.batch_size], device=device)
            output = torch.maximum(base_tensor[indices]+model(inputs[indices]), -mean[indices]/std[indices])
            raw = torch.expm1(output*std[indices]+mean[indices])
            selected = mask_tensor[indices]
            loss = ((raw-target[indices]).abs()/scale[indices])[selected].mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite masked training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            points = int(selected.sum())
            total += float(loss.detach())*points
            count += points
     
        score = None
        if epoch % 5 == 0 or epoch == p.epochs:
            value = _predict(model, features, validation, scaler, validation_base, device)
            score = error_blocks(value, validation, scaler)["score"]
            if score < best_score:
                best_score, best_epoch, best = score, epoch, copy.deepcopy(model.state_dict())
      
        history.append(dict(epoch=epoch, train_loss=total/count, validation_score=score))
        if epoch % 20 == 0:
            print(f"epoch={epoch} init={seed} lr={lr} train={total/count:.6f} val={score:.6f}", flush=True)
   
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
  
    seconds = time.perf_counter()-begin
    model.load_state_dict(best)
    model.eval()
    return model, dict(seed=seed, learning_rate=lr, parameters=sum(v.numel() for v in model.parameters()), epochs=p.epochs,
                       best_epoch=best_epoch, initial_validation=initial, validation_score=best_score,
                       seconds=seconds, history=history)
