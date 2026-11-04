"""Training with fixed checkpoint criterion and sample-weighted metrics."""
import math
import time
from dataclasses import dataclass, field
import numpy as np
import torch
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
    selection_loss: list = field(default_factory=list)
    best_epoch: int | None = None
    best_beta: float | None = None
    best_score: float | None = None


class EarlyStopping:
    def __init__(self, patience, min_delta):
        if patience < 1 or min_delta < 0:
            raise ValueError("invalid early stopping settings")
        self.patience, self.min_delta = patience, min_delta
        self.best = float("inf")
        self.bad_epochs = 0

    def should_stop(self, val_loss):
        if not math.isfinite(val_loss):
            raise FloatingPointError("Nonfinite validation loss")
        if val_loss < self.best - self.min_delta:
            self.best, self.bad_epochs = val_loss, 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def seed_everything(seed, deterministic=False):
    import random
    import os
    if deterministic:
        # Must be set before the first cuBLAS call in this process.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(deterministic)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _run_epoch(model, loader, cfg, beta, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    totals = dict(loss=0., recon=0., kl=0., endpoint_mae=0.)
    n_samples = 0
    for index, batch in enumerate(loader):
        paired = isinstance(batch, (tuple, list))
        target_kind = cfg.training_target if is_train else cfg.validation_target
        if target_kind == "clean" and not paired:
            raise ValueError("Clean reconstruction requires paired input/target batches")
        if paired and len(batch) != 2:
            raise ValueError("Expected an input/target pair")
        x, target = (batch[0].to(cfg.device), batch[1].to(cfg.device)) if paired else (batch.to(cfg.device), batch.to(cfg.device))
        if x.shape != target.shape or not torch.isfinite(x).all() or not torch.isfinite(target).all():
            raise FloatingPointError(f"Invalid input/target in batch {index}")
        with torch.set_grad_enabled(is_train):
            if is_train and cfg.teacher_forcing:
                x_hat, _, mu, logvar = model(x, use_teacher_forcing=True)
            else:
                x_hat, _, mu, logvar = model(x)
            loss, parts = compute_loss(x_hat, target, mu, logvar, model.variational, cfg.recon_loss, beta)
            parts["endpoint_mae"] = (x_hat[:, -1]-target[:, -1]).abs().mean().item()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss in batch {index}; inspect inputs and logvar")
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm or float("inf"),
                                          error_if_nonfinite=True)
            optimizer.step()
        n_samples += len(x)
        for key in totals:
            totals[key] += parts[key] * len(x)
    if not n_samples:
        raise ValueError("Empty loader: no samples processed")
    return {key: value / n_samples for key, value in totals.items()}


def train_model(model, train_loader, val_loader, cfg, verbose=True):
    cfg.__post_init__()
    if model.variational and cfg.epochs <= cfg.kl_anneal_epochs:
        raise ValueError("VAE epochs must exceed warm-up to reach beta_target")
    if cfg.teacher_forcing and not getattr(model.decoder, "supports_teacher_forcing", False):
        raise ValueError("teacher_forcing requires an LSTM decoder")
    seed_everything(cfg.seed, cfg.deterministic)
    model.to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    annealer = KLAnnealer(cfg.beta_target, cfg.kl_anneal_epochs)
    stopper = EarlyStopping(cfg.early_stopping_patience, cfg.early_stopping_min_delta)
    history, best_state, best_val = History(), None, float("inf")
    for epoch in range(cfg.epochs):
        start = time.perf_counter()
        beta = annealer.beta(epoch) if model.variational else 0.
        train = _run_epoch(model, train_loader, cfg, beta, optimizer)
        val = _run_epoch(model, val_loader, cfg, beta)
        # Deterministic reconstruction at mu, NOT a Monte Carlo ELBO estimate.
        score = (val["endpoint_mae"] if cfg.selection_metric == "endpoint_mae" else
                 val["recon"] + (cfg.beta_target * val["kl"] if cfg.selection_metric == "fixed_beta" else 0.))
        for key, value in (("train_loss", train["loss"]), ("val_loss", val["loss"]),
                           ("train_recon", train["recon"]), ("val_recon", val["recon"]),
                           ("val_kl", val["kl"]), ("beta", beta), ("selection_loss", score),
                           ("epoch_time_sec", time.perf_counter()-start)):
            getattr(history, key).append(value)
        eligible = not model.variational or epoch >= cfg.kl_anneal_epochs
        if eligible and score < best_val:
            best_val = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            history.best_epoch, history.best_beta, history.best_score = epoch+1, beta, score
        if verbose:
            print(f"epoch {epoch+1}/{cfg.epochs} train={train['loss']:.5f} val_recon={val['recon']:.5f} "
                  f"KL={val['kl']:.5f} beta={beta:.4g} selection={score:.5f}", flush=True)
        if eligible and stopper.should_stop(score):
            break
    if best_state is None:
        raise RuntimeError("No eligible checkpoint")
    model.load_state_dict(best_state)
    return history
