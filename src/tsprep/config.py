"""
Конфигурации эксперимента.

Все гиперпараметры, упомянутые в отчёте (T, dz, beta, K эпох KL-annealing,
порог градиентного клиппинга C, порог ранней остановки delta и т.д.),
собраны здесь как явные, документированные поля -- вместо "магических
чисел" внутри кода.
"""
from dataclasses import dataclass, field
from typing import Literal, Optional

ModelName = Literal[
    "tcn_ae", "tcn_vae",
    "lstm_ae", "lstm_vae",
    "transformer_ae",
    "hybrid_vae",  # TCN + Attention, предложенная модель (VAE-регуляризация)
] | str


@dataclass
class DataConfig:
    # Длина окна T и признаки F (глава 2.1 отчёта: X in R^{B x T x F})
    window_len: int = 64
    stride: int = 1
    n_features: int = 1
    # Горизонт прогноза для downstream-задачи прогнозирования (глава 3.3)
    forecast_horizon: int = 5
    # Хронологическое разбиение 70/15/15, БЕЗ перемешивания (глава 3.1)
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15
    batch_size: int = 64
    # Детрендирование: обычная первая разность или сезонная (period>1)
    seasonal_period: int = 1
    # Логарифмирование перед дифференцированием.
    # Рекомендуется ставить True для абсолютных ценовых
    # рядов, охватывающих много лет (иначе train/val нормализация
    # "разъезжается" из-за роста цены со временем).
    log_transform: bool = True
    seed: int = 42


@dataclass
class ModelConfig:
    model_name: ModelName = "hybrid_vae"
    latent_dim: int = 16          # d_z
    hidden_dim: int = 64          # каналы/скрытая размерность бэкбона
    dropout: float = 0.1

    # TCN-специфичные параметры (глава 1.2, 2.2)
    tcn_levels: int = 5           # число слоёв => receptive field ~ 2^levels
    tcn_kernel_size: int = 3

    # LSTM-специфичные параметры
    lstm_num_layers: int = 2

    # Transformer / Hybrid-attention параметры
    n_heads: int = 4
    n_attn_layers: int = 2
    d_model: int = 64

    @property
    def variational(self) -> bool:
        return self.model_name in ("tcn_vae", "lstm_vae", "hybrid_vae")

    @property
    def backbone(self) -> str:
        return {
            "tcn_ae": "tcn", "tcn_vae": "tcn",
            "lstm_ae": "lstm", "lstm_vae": "lstm",
            "transformer_ae": "transformer",
            "hybrid_vae": "hybrid",
        }[self.model_name]


@dataclass
class TrainConfig:
    epochs: int = 60
    lr: float = 1e-3
    weight_decay: float = 1e-4     # λ в AdamW (глава 3.2)
    recon_loss: Literal["mse", "mae"] = "mse"

    # KL-annealing (глава 2.4 / 3.2): beta растёт линейно 0 -> beta_target
    # в течение kl_anneal_epochs эпох.
    beta_target: float = 1.0
    kl_anneal_epochs: int = 15

    # Градиентный клиппинг (глава 3.2): порог C=1.0
    grad_clip_norm: Optional[float] = 1.0

    # Критерий ранней остановки (глава 3.2):
    # Stop <=> min(L_val[t-P:T]) - L_val(t) < delta
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4

    device: str = "cpu"
    seed: int = 42


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


ALL_MODEL_NAMES = [
    "tcn_ae", "tcn_vae", "lstm_ae", "lstm_vae", "transformer_ae", "hybrid_vae",
]