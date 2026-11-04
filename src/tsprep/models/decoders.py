"""Декодеры z (B,d_z) в окно (B,T,F). TCN и Hybrid используют общий декодер."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from tsprep.models.layers import PositionalEncoding, TCNBackbone


class TCNDecoder(nn.Module):
    def __init__(self, n_features, hidden_dim, latent_dim, window_len, n_levels, kernel_size, dropout,
                 upsampling="matmul"):
        super().__init__()
        self.window_len = window_len
        self.seed_len = max(4, window_len // (2 ** min(n_levels, 4)))
        self.upsampling = upsampling
        # Same linear map as align_corners=False interpolation, built on CPU.
        # GEMM backward avoids CUDA interpolation's atomic gradient accumulation.
        basis = torch.eye(self.seed_len).unsqueeze(0)
        matrix = F.interpolate(basis, size=window_len, mode="linear", align_corners=False)[0]
        self.register_buffer("interpolation_matrix", matrix, persistent=False)
        self.fc_seed = nn.Linear(latent_dim, hidden_dim * self.seed_len)
        self.hidden_dim = hidden_dim
        self.tcn = TCNBackbone(hidden_dim, hidden_dim, n_levels, kernel_size, dropout)
        self.out_proj = nn.Conv1d(hidden_dim, n_features, kernel_size=1)

    def forward(self, z):
        b = z.shape[0]
        seed = self.fc_seed(z).view(b, self.hidden_dim, self.seed_len)
        upsampled = (seed @ self.interpolation_matrix if self.upsampling == "matmul" else
                     F.interpolate(seed, size=self.window_len, mode="linear", align_corners=False))
        h = self.tcn(upsampled.transpose(1, 2))    # TCNBackbone expects (B,T,F)-like input
        x_hat = self.out_proj(h).transpose(1, 2)    # (B, T, F)
        return x_hat


class LSTMDecoder(nn.Module):
    supports_teacher_forcing = True

    def __init__(self, n_features, hidden_dim, latent_dim, window_len, num_layers, dropout):
        super().__init__()
        self.window_len = window_len
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.fc_init = nn.Linear(latent_dim, hidden_dim * num_layers)
        self.lstm_cell = nn.LSTM(
            input_size=n_features, hidden_size=hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_proj = nn.Linear(hidden_dim, n_features)

    def forward(self, z, teacher_forcing_target=None):
        b = z.shape[0]
        h0 = torch.tanh(self.fc_init(z)).view(b, self.num_layers, self.hidden_dim)
        h0 = h0.permute(1, 0, 2).contiguous()
        c0 = torch.zeros_like(h0)

        if teacher_forcing_target is not None and self.training:
            if teacher_forcing_target.shape != (b, self.window_len, self.n_features):
                raise ValueError("teacher forcing target has incorrect shape")
            shifted = torch.cat([z.new_zeros(b, 1, self.n_features), teacher_forcing_target[:, :-1]], dim=1)
            out, _ = self.lstm_cell(shifted, (h0, c0))
            return self.out_proj(out)

        outputs = []
        # первый вход декодера - нулевой вектор ("start token")
        step_input = z.new_zeros(b, 1, self.n_features)
        hidden = (h0, c0)
        for t in range(self.window_len):
            out, hidden = self.lstm_cell(step_input, hidden)
            pred = self.out_proj(out)              # (B, 1, F)
            outputs.append(pred)
            if teacher_forcing_target is not None and self.training:
                step_input = teacher_forcing_target[:, t:t + 1, :]
            else:
                step_input = pred
        return torch.cat(outputs, dim=1)            # (B, T, F)


class TransformerDecoder(nn.Module):
    """Параллельная реконструкция по z и позиционным векторам."""

    def __init__(self, n_features, d_model, latent_dim, window_len, n_heads, n_layers, dropout):
        super().__init__()
        self.window_len = window_len
        self.z_proj = nn.Linear(latent_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_proj = nn.Linear(d_model, n_features)

    def forward(self, z):
        b = z.shape[0]
        seed = self.z_proj(z).unsqueeze(1).repeat(1, self.window_len, 1)  # (B,T,d_model)
        h = self.pos_enc(seed)
        h = self.decoder(h)
        return self.out_proj(h)


def build_decoder(model_cfg, n_features: int, window_len: int) -> nn.Module:
    m = model_cfg
    if m.backbone in ("tcn", "hybrid"):
        return TCNDecoder(n_features, m.hidden_dim, m.latent_dim, window_len,
                           m.tcn_levels, m.tcn_kernel_size, m.dropout, m.upsampling)
    if m.backbone == "lstm":
        return LSTMDecoder(n_features, m.hidden_dim, m.latent_dim, window_len,
                            m.lstm_num_layers, m.dropout)
    if m.backbone == "transformer":
        return TransformerDecoder(n_features, m.d_model, m.latent_dim, window_len,
                                   m.n_heads, m.n_attn_layers, m.dropout)
    raise ValueError(f"Unknown backbone: {m.backbone}")
