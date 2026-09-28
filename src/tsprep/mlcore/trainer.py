"""
Цикл обучения:
  - оптимизатор AdamW(lr, weight_decay),
  - градиентный клиппинг ||grad|| <= C,
  - KL-annealing для VAE-вариантов,
  - ранняя остановка по критерию:
        Stop <=> min(L_val[t-P : t]) - L_val(t) < delta
    (то есть валидационная потеря не улучшается сильнее, чем на delta,
    на протяжении последних P эпох).
"""
import copy
import time
from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader

from tsprep.config import TrainConfig
from tsprep.models.autoencoder import Autoencoder
from tsprep.mlcore.losses import KLAnnealer, compute_loss


@dataclass
class History:
    train_loss: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    train_recon: list = field(default_factory=list)
    val_recon: list = field(default_factory=list)
    val_kl: list = field(default_factory=list)
    beta: list = field(default_factory=list)
    epoch_time_sec: list = field(default_factory=list)


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float):
        self.patience = patience
        self.min_delta = min_delta
        self.history: list[float] = []

    def should_stop(self, val_loss: float) -> bool:
        self.history.append(val_loss)
        if len(self.history) <= self.patience:
            return False
        window = self.history[-self.patience - 1:-1]
        improvement = min(window) - val_loss
        return improvement < self.min_delta


def _run_epoch(model: Autoencoder, loader: DataLoader, cfg: TrainConfig, beta: float,
               optimizer=None) -> dict:
    is_train = optimizer is not None
    model.train(is_train)
    totals = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
    n_batches = 0
    for batch in loader:
        x = batch.to(cfg.device)
        with torch.set_grad_enabled(is_train):
            x_hat, z, mu, logvar = model(x)
            loss, parts = compute_loss(x_hat, x, mu, logvar, model.variational, cfg.recon_loss, beta)
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            if cfg.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()
        totals["loss"] += parts["loss"]
        totals["recon"] += parts["recon"]
        totals["kl"] += parts["kl"]
        n_batches += 1
    n_batches = max(1, n_batches)
    return {k: v / n_batches for k, v in totals.items()}


def train_model(
    model: Autoencoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    verbose: bool = True,
) -> History:
    model.to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    annealer = KLAnnealer(cfg.beta_target, cfg.kl_anneal_epochs)
    stopper = EarlyStopping(cfg.early_stopping_patience, cfg.early_stopping_min_delta)

    history = History()
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")

    for epoch in range(cfg.epochs):
        t0 = time.time()
        beta = annealer.beta(epoch) if model.variational else 0.0

        train_stats = _run_epoch(model, train_loader, cfg, beta, optimizer=optimizer)
        val_stats = _run_epoch(model, val_loader, cfg, beta, optimizer=None)
        dt = time.time() - t0

        history.train_loss.append(train_stats["loss"])
        history.val_loss.append(val_stats["loss"])
        history.train_recon.append(train_stats["recon"])
        history.val_recon.append(val_stats["recon"])
        history.val_kl.append(val_stats["kl"])
        history.beta.append(beta)
        history.epoch_time_sec.append(dt)

        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            best_state = copy.deepcopy(model.state_dict())

        if verbose:
            print(f"epoch {epoch+1:3d}/{cfg.epochs} | "
                  f"train_loss={train_stats['loss']:.4f} val_loss={val_stats['loss']:.4f} "
                  f"val_recon={val_stats['recon']:.4f} val_kl={val_stats['kl']:.4f} "
                  f"beta={beta:.3f} ({dt:.1f}s)")

        if stopper.should_stop(val_stats["loss"]):
            if verbose:
                print(f"Early stopping at epoch {epoch+1} "
                      f"(no improvement > {cfg.early_stopping_min_delta} over last "
                      f"{cfg.early_stopping_patience} epochs)")
            break

    model.load_state_dict(best_state)
    return history