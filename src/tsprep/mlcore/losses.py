"""Потери реконструкции, KL-дивергенция и линейный разогрев beta."""
import math
import torch
import torch.nn.functional as F


def reconstruction_loss(x_hat: torch.Tensor, x: torch.Tensor, kind: str = "mse") -> torch.Tensor:
    if kind == "mse":
        return F.mse_loss(x_hat, x, reduction="mean")
   
    if kind == "mae":
        return F.l1_loss(x_hat, x, reduction="mean")
   
    raise ValueError(f"Unknown recon loss kind: {kind}")


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    # среднее по батчу суммы по латентным измерениям
    kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return kl_per_sample.mean()


def residual_loss(prediction, target, kind="mse", *, clean_prediction=None, identity_weight=0.):
    """Endpoint denoising loss; identity term compares f(clean) with the same target."""
    if not math.isfinite(identity_weight) or identity_weight < 0:
        raise ValueError("identity weight must be finite and nonnegative")
  
    if kind not in ("mae", "mse", "mse_identity"):
        raise ValueError(f"Unknown residual loss: {kind}")
  
    if prediction.shape != target.shape or target.ndim != 1:
        raise ValueError("expected aligned endpoint vectors")
 
    if kind != "mse_identity" and identity_weight:
        raise ValueError("identity weight requires mse_identity")
 
    error = prediction-target
    loss = error.abs().mean() if kind == "mae" else error.square().mean()
    if kind == "mse_identity" and identity_weight:
        if clean_prediction is None or clean_prediction.shape != target.shape:
            raise ValueError("identity penalty requires aligned clean predictions")
  
        loss = loss + identity_weight*(clean_prediction-target).square().mean()
  
    return loss


class KLAnnealer:
    """beta(epoch) = beta_target * min(1, epoch / kl_anneal_epochs)."""

    def __init__(self, beta_target: float, kl_anneal_epochs: int):
        self.beta_target = beta_target
        self.kl_anneal_epochs = kl_anneal_epochs

    def beta(self, epoch: int) -> float:
        if self.kl_anneal_epochs == 0:
            return self.beta_target
        return self.beta_target * min(1.0, epoch / self.kl_anneal_epochs)


def compute_loss(x_hat, x, mu, logvar, variational: bool, recon_kind: str, beta: float):
    recon = reconstruction_loss(x_hat, x, recon_kind)
    if variational:
        kl = kl_divergence(mu, logvar)
        total = recon + beta * kl
        return total, {"loss": total.item(), "recon": recon.item(), "kl": kl.item(), "beta": beta}
    total = recon
    return total, {"loss": total.item(), "recon": recon.item(), "kl": 0.0, "beta": 0.0}
