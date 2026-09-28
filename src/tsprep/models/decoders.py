"""
Декодеры z (B,d_z) -> X_hat (B,T,F).

  - TCNDecoder: z проецируется в короткую "затравку", растягивается
    интерполяцией до длины T, затем каузальный TCN восстанавливает
    временную структуру (гл. 2.2 отчета - декодер, обратный TCN-энкодеру).
  - LSTMDecoder: z инициализирует скрытое состояние LSTM, авторегрессивная
    генерация (teacher forcing на обучении) - покомпонентное восстановление.
  - TransformerDecoder: z повторяется как псевдо-последовательность
    запросов, self-attention восстанавливает X_hat "непоследовательно"
    (без авторегрессии, быстрее в обучении/инференсе).

Для гибридной (TCN+Attention) модели используется тот же декодер, что
и у TCN-варианта - асимметричная архитектура энкодер/декодер обсуждена
в отчете как допустимая, поскольку декодер отвечает только за
реконструкцию, а не за извлечение признаков.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from tsprep.models.layers import PositionalEncoding, TCNBackbone


class TCNDecoder(nn.Module):
    def __init__(self, n_features, hidden_dim, latent_dim, window_len, n_levels, kernel_size, dropout):
        super().__init__()
        self.window_len = window_len
        self.seed_len = max(4, window_len // (2 ** min(n_levels, 4)))
        self.fc_seed = nn.Linear(latent_dim, hidden_dim * self.seed_len)
        self.hidden_dim = hidden_dim
        self.tcn = TCNBackbone(hidden_dim, hidden_dim, n_levels, kernel_size, dropout)
        self.out_proj = nn.Conv1d(hidden_dim, n_features, kernel_size=1)

    def forward(self, z):
        b = z.shape[0]
        seed = self.fc_seed(z).view(b, self.hidden_dim, self.seed_len)
        upsampled = F.interpolate(seed, size=self.window_len, mode="linear", align_corners=False)
        h = self.tcn(upsampled.transpose(1, 2))    # TCNBackbone expects (B,T,F)-like input
        x_hat = self.out_proj(h).transpose(1, 2)    # (B, T, F)
        return x_hat


class LSTMDecoder(nn.Module):
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
        h0 = torch.tanh(self.fc_init(z)).view(self.num_layers, b, self.hidden_dim).contiguous()
        c0 = torch.zeros_like(h0)

        outputs = []
        # первый вход декодера - нулевой вектор ("start token")
        step_input = torch.zeros(b, 1, self.n_features, device=z.device)
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
    """Непоследовательная (non-autoregressive) реконструкция: z повторяется
    как T query-векторов, self-attention учится расставить их во времени
    с помощью позиционного кодирования."""

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
                           m.tcn_levels, m.tcn_kernel_size, m.dropout)
    if m.backbone == "lstm":
        return LSTMDecoder(n_features, m.hidden_dim, m.latent_dim, window_len,
                            m.lstm_num_layers, m.dropout)
    if m.backbone == "transformer":
        return TransformerDecoder(n_features, m.d_model, m.latent_dim, window_len,
                                   m.n_heads, m.n_attn_layers, m.dropout)
    raise ValueError(f"Unknown backbone: {m.backbone}")