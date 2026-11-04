"""Сборка AE/VAE из энкодера и декодера по ModelConfig."""
import torch
import torch.nn as nn

from tsprep.config import ModelConfig
from tsprep.models.decoders import build_decoder
from tsprep.models.encoders import build_encoder


class Autoencoder(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, variational: bool):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.variational = variational

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """z = mu + sigma * eps,  eps ~ N(0, I)  (reparametrization trick)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor):
        mu_or_z, logvar = self.encoder(x)
        if self.variational:
            z = self.reparameterize(mu_or_z, logvar) if self.training else mu_or_z
            return z, mu_or_z, logvar
        return mu_or_z, mu_or_z, None

    def forward(self, x: torch.Tensor, use_teacher_forcing: bool = False):
        z, mu, logvar = self.encode(x)
        if use_teacher_forcing and not getattr(self.decoder, "supports_teacher_forcing", False):
            raise ValueError("Teacher forcing is supported only by LSTMDecoder")
        if use_teacher_forcing and self.training:
            x_hat = self.decoder(z, teacher_forcing_target=x)
        else:
            x_hat = self.decoder(z)
                
        return x_hat, z, mu, logvar


def build_model(model_cfg: ModelConfig, n_features: int, window_len: int) -> Autoencoder:
    if n_features < 1 or window_len < 1:
        raise ValueError("n_features and window_len must be positive")
    model_cfg.__post_init__()
    encoder = build_encoder(model_cfg, n_features)
    decoder = build_decoder(model_cfg, n_features, window_len)
    return Autoencoder(encoder, decoder, variational=model_cfg.variational)
