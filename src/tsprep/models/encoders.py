"""
Энкодеры X (B,T,F) -> z (B,d_z), либо (mu, logvar) для VAE-варианта.

Реализованы все архитектуры, сравниваемые в отчете (гл. 2, 3.3):
  - TCNEncoder            (backbone: каузальный TCN)
  - LSTMEncoder            (backbone: многослойный LSTM)
  - TransformerEncoder     (backbone: self-attention энкодер)
  - HybridTCNAttentionEncoder - предложенная модель: TCN извлекает
    локальные многомасштабные признаки, self-attention поверх них
    моделирует глобальные зависимости (гл. 1.3 / 2.3 отчета).
"""
import torch
import torch.nn as nn

from tsprep.models.layers import PositionalEncoding, TCNBackbone


class _BottleneckHead(nn.Module):
    """Общая "голова": pooled-представление -> z (или mu, logvar)."""

    def __init__(self, in_dim: int, latent_dim: int, variational: bool):
        super().__init__()
        self.variational = variational
        if variational:
            self.fc_mu = nn.Linear(in_dim, latent_dim)
            self.fc_logvar = nn.Linear(in_dim, latent_dim)
        else:
            self.fc_z = nn.Linear(in_dim, latent_dim)

    def forward(self, pooled):
        if self.variational:
            mu = self.fc_mu(pooled)
            logvar = self.fc_logvar(pooled)
            return mu, logvar
        return self.fc_z(pooled), None


class TCNEncoder(nn.Module):
    def __init__(self, n_features, hidden_dim, latent_dim, n_levels, kernel_size, dropout, variational):
        super().__init__()
        self.tcn = TCNBackbone(n_features, hidden_dim, n_levels, kernel_size, dropout)
        self.head = _BottleneckHead(hidden_dim, latent_dim, variational)

    def forward(self, x):
        h = self.tcn(x)                      # (B, C, T)
        pooled = h.mean(dim=2)                # global average pooling по времени
        mu_or_z, logvar = self.head(pooled)
        return mu_or_z, logvar


class LSTMEncoder(nn.Module):
    def __init__(self, n_features, hidden_dim, latent_dim, num_layers, dropout, variational):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = _BottleneckHead(hidden_dim, latent_dim, variational)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]                 # (B, hidden_dim) - состояние последнего слоя
        mu_or_z, logvar = self.head(last_hidden)
        return mu_or_z, logvar


class TransformerEncoder(nn.Module):
    def __init__(self, n_features, d_model, latent_dim, n_heads, n_layers, dropout, variational):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = _BottleneckHead(d_model, latent_dim, variational)

    def forward(self, x):
        h = self.pos_enc(self.input_proj(x))
        h = self.encoder(h)                   # (B, T, d_model)
        pooled = h.mean(dim=1)                # усреднение по времени вместо CLS-токена
        mu_or_z, logvar = self.head(pooled)
        return mu_or_z, logvar


class HybridTCNAttentionEncoder(nn.Module):
    """
    Предложенная модель (гл. 1.3, 2.3 отчета): TCN как экстрактор
    локальных многомасштабных признаков + multi-head self-attention
    поверх последовательности TCN-признаков для захвата долгосрочных
    зависимостей, которые ограниченный receptive field TCN не видит
    напрямую.
    """

    def __init__(self, n_features, hidden_dim, latent_dim, n_levels, kernel_size,
                 n_heads, n_attn_layers, dropout, variational):
        super().__init__()
        self.tcn = TCNBackbone(n_features, hidden_dim, n_levels, kernel_size, dropout)
        self.pos_enc = PositionalEncoding(hidden_dim)
        attn_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=4 * hidden_dim,
            dropout=dropout, batch_first=True,
        )
        self.attn = nn.TransformerEncoder(attn_layer, num_layers=n_attn_layers)
        # Обучаемое внимание-пулинг (взвешенное среднее по времени)
        self.pool_query = nn.Linear(hidden_dim, 1)
        self.head = _BottleneckHead(hidden_dim, latent_dim, variational)

    def forward(self, x):
        local_feats = self.tcn(x).transpose(1, 2)     # (B, T, hidden_dim) - локальные признаки TCN
        h = self.pos_enc(local_feats)
        h = self.attn(h)                               # (B, T, hidden_dim) - глобальный контекст
        attn_weights = torch.softmax(self.pool_query(h), dim=1)  # (B, T, 1)
        pooled = (h * attn_weights).sum(dim=1)          # (B, hidden_dim) attention pooling
        mu_or_z, logvar = self.head(pooled)
        return mu_or_z, logvar


def build_encoder(model_cfg, n_features: int) -> nn.Module:
    m = model_cfg
    if m.backbone == "tcn":
        return TCNEncoder(n_features, m.hidden_dim, m.latent_dim, m.tcn_levels,
                           m.tcn_kernel_size, m.dropout, m.variational)
    if m.backbone == "lstm":
        return LSTMEncoder(n_features, m.hidden_dim, m.latent_dim, m.lstm_num_layers,
                            m.dropout, m.variational)
    if m.backbone == "transformer":
        return TransformerEncoder(n_features, m.d_model, m.latent_dim, m.n_heads,
                                   m.n_attn_layers, m.dropout, m.variational)
    if m.backbone == "hybrid":
        return HybridTCNAttentionEncoder(n_features, m.hidden_dim, m.latent_dim, m.tcn_levels,
                                          m.tcn_kernel_size, m.n_heads, m.n_attn_layers,
                                          m.dropout, m.variational)
    raise ValueError(f"Unknown backbone: {m.backbone}")