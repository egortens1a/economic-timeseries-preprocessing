"""Исправление ошибок наблюдения по доступной истории с сохранением длины окна."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from tsprep.models.layers import TCNBackbone


class ResidualTCNDenoiser(nn.Module):
    """Start at identity; estimate an additive error at every time position."""

    def __init__(self, n_features=1, hidden_dim=16, levels=3, kernel_size=3, local_context=0,
                 context_mode="local", correction_delay=0, delay_mode="next"):
        super().__init__()
        if min(n_features, hidden_dim, levels, kernel_size) < 1:
            raise ValueError("model dimensions must be positive")
       
        if not isinstance(local_context, int) or local_context < 0:
            raise ValueError("local context must be a nonnegative integer")
     
        if context_mode not in ("local", "repeat") or (context_mode == "repeat" and not local_context):
            raise ValueError("repeat control requires a positive local context")
     
        if correction_delay not in (0, 1) or delay_mode not in ("past", "next"):
            raise ValueError("unsupported correction delay or delay mode")
     
        if correction_delay and local_context:
            raise ValueError("delay experiment excludes local context features")
    
        if not correction_delay and delay_mode != "next":
            raise ValueError("past control requires one-step delay")
    
        self.local_context = local_context
        self.context_mode = context_mode
        self.correction_delay, self.delay_mode = correction_delay, delay_mode
        channels = 2*n_features if correction_delay else (4*n_features if local_context else n_features)
        self.backbone = TCNBackbone(channels, hidden_dim, levels, kernel_size, 0.)
        self.error_head = nn.Conv1d(hidden_dim, n_features, 1)
        nn.init.zeros_(self.error_head.weight)
        nn.init.zeros_(self.error_head.bias)

    def _local_features(self, x):
        """Only previous observations determine location/scale; early history is zero padded."""
        history = F.pad(x.transpose(1, 2), (self.local_context, 0))
        history = history.unfold(2, self.local_context, 1)[:, :, :-1]
        # Стабильная сортировка сохраняет выбор нижней медианы и воспроизводимость на CUDA.
        middle = (self.local_context-1)//2
        center = history.sort(dim=-1, stable=True).values[..., middle]
        mad = (history-center.unsqueeze(-1)).abs().sort(dim=-1, stable=True).values[..., middle]
        scale = (1.4826*mad).clamp_min(1e-3)
        center, scale = center.transpose(1, 2), scale.transpose(1, 2)
        deviation = ((x-center)/scale).clamp(-20., 20.)
        return torch.cat((x, center, scale.log(), deviation), dim=-1)

    def reference(self, x):
        """Observed value being repaired at each availability time."""
        return torch.cat((torch.zeros_like(x[:, :1]), x[:, :-1]), dim=1) if self.correction_delay else x

    def forward(self, x):
        reference = self.reference(x)
        features = x
        if self.correction_delay:
            features = torch.cat((reference, x if self.delay_mode == "next" else reference), dim=-1)
        elif self.local_context:
            features = torch.cat((x, x, x, x), dim=-1) if self.context_mode == "repeat" else self._local_features(x)
        correction = self.error_head(self.backbone(features)).transpose(1, 2)
        return reference - correction
