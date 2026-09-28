"""
Базовые строительные блоки: каузальная дилатированная свертка (TCN),
позиционное кодирование для трансформера.

Формулы соответствуют гл. 1.2 / 2.2 отчета:
  y_t^(i) = sum_{k=0}^{K-1} w_k^(i) * x^(i-1)_{t - d_i * k}
  d_i = 2^i (экспоненциальный рост дилатации)
  skip: o^(i) = ReLU(Conv(x^(i-1)) + x^(i-1))
"""
import math

import torch
import torch.nn as nn


class Chomp1d(nn.Module):
    """Обрезает "лишний" правый паддинг, обеспечивая каузальность (без
    заглядывания в будущее): выход в момент t зависит только от t' <= t."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Один уровень TCN: две каузальные дилатированные свертки + residual/skip."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        # 1x1 проекция для skip-соединения, если каналы не совпадают
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu_out = nn.ReLU()

    def forward(self, x):
        # x: (B, C, T)
        out = self.drop1(self.relu1(self.chomp1(self.conv1(x))))
        out = self.drop2(self.relu2(self.chomp2(self.conv2(out))))
        residual = x if self.downsample is None else self.downsample(x)
        return self.relu_out(out + residual)


class TCNBackbone(nn.Module):
    """Стек TemporalBlock с дилатацией d_i = 2^i, растущей по глубине."""

    def __init__(self, n_features: int, hidden_dim: int, n_levels: int, kernel_size: int, dropout: float):
        super().__init__()
        layers = []
        in_ch = n_features
        for i in range(n_levels):
            dilation = 2 ** i
            layers.append(TemporalBlock(in_ch, hidden_dim, kernel_size, dilation, dropout))
            in_ch = hidden_dim
        self.network = nn.Sequential(*layers)
        self.out_channels = hidden_dim

    def forward(self, x):
        # x: (B, T, F) -> conv expects (B, C, T)
        h = x.transpose(1, 2)
        h = self.network(h)
        return h  # (B, hidden_dim, T)


class PositionalEncoding(nn.Module):
    """Стандартное синусоидальное позиционное кодирование."""

    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, T, d_model)
        return x + self.pe[:, : x.size(1)] # type: ignore