"""TCN с двумя направлениями обработки окна для заполнения пропусков."""
import torch
from torch import nn
from tsprep.models.layers import TCNBackbone


class MaskedTCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = TCNBackbone(5, 16, 3, 3, 0.)
        self.head = nn.Conv1d(32, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        forward = self.backbone(x)
        reverse = self.backbone(x.flip(1)).flip(2)
        return self.head(torch.cat([forward, reverse], dim=1))[:, 0]

