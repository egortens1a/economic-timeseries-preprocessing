"""
Функции потерь и планировщик KL-annealing

L_VAE = L_recon + beta * D_KL(q(z|x) || N(0, I))

D_KL(N(mu,sigma^2) || N(0,1)) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))

beta растет линейно от 0 до beta_target в течение kl_anneal_epochs -
это стандартный прием против "коллапса апостериорного распределения"
(posterior collapse), когда модель игнорирует z и декодер работает
как обычный AE
"""
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


class KLAnnealer:
    """beta(epoch) = beta_target * min(1, epoch / kl_anneal_epochs)."""

    def __init__(self, beta_target: float, kl_anneal_epochs: int):
        self.beta_target = beta_target
        self.kl_anneal_epochs = max(1, kl_anneal_epochs)

    def beta(self, epoch: int) -> float:
        return self.beta_target * min(1.0, epoch / self.kl_anneal_epochs)


def compute_loss(x_hat, x, mu, logvar, variational: bool, recon_kind: str, beta: float):
    recon = reconstruction_loss(x_hat, x, recon_kind)
    if variational:
        kl = kl_divergence(mu, logvar)
        total = recon + beta * kl
        return total, {"loss": total.item(), "recon": recon.item(), "kl": kl.item(), "beta": beta}
    total = recon
    return total, {"loss": total.item(), "recon": recon.item(), "kl": 0.0, "beta": 0.0}