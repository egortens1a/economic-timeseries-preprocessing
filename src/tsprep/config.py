"""Serializable, validated experiment configuration."""
from dataclasses import dataclass, field
import math

ALL_MODEL_NAMES = ["tcn_ae", "tcn_vae", "lstm_ae", "lstm_vae", "transformer_ae", "hybrid_vae"]
ABLATION_MODEL_NAMES = ["hybrid_ae", "transformer_vae"]


def positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass
class DataConfig:
    window_len: int = 64
    stride: int = 1
    n_features: int = 1
    forecast_horizon: int = 5
    target_feature: int = 0
    train_frac: float = .70
    val_frac: float = .15
    test_frac: float = .15
    batch_size: int = 64
    seasonal_period: int = 1
    log_transform: bool = True
    seed: int = 42

    def __post_init__(self):
        for name in ("window_len", "stride", "n_features", "forecast_horizon", "batch_size"):
            positive_int(name, getattr(self, name))
        if not isinstance(self.seasonal_period, int) or self.seasonal_period < 0:
            raise ValueError("seasonal_period must be a nonnegative integer")
        if not isinstance(self.target_feature, int) or not 0 <= self.target_feature < self.n_features:
            raise ValueError("target_feature outside feature range")
        fractions = (self.train_frac, self.val_frac, self.test_frac)
        if any(not 0 < x < 1 for x in fractions) or not math.isclose(sum(fractions), 1., abs_tol=1e-8):
            raise ValueError("positive train/val/test fractions must sum to 1")


@dataclass
class ModelConfig:
    model_name: str = "hybrid_vae"
    latent_dim: int = 16
    hidden_dim: int = 64
    dropout: float = .1
    tcn_levels: int = 5
    tcn_kernel_size: int = 3
    lstm_num_layers: int = 2
    n_heads: int = 4
    n_attn_layers: int = 2
    d_model: int = 64
    upsampling: str = "matmul"

    def __post_init__(self):
        if self.model_name not in ALL_MODEL_NAMES + ABLATION_MODEL_NAMES:
            raise ValueError(f"Unknown model: {self.model_name}")
        for name in ("latent_dim", "hidden_dim", "tcn_levels", "tcn_kernel_size", "lstm_num_layers",
                     "n_heads", "n_attn_layers", "d_model"):
            positive_int(name, getattr(self, name))
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.upsampling not in ("linear", "matmul"):
            raise ValueError("upsampling must be linear or matmul")
        width = self.hidden_dim if self.backbone == "hybrid" else self.d_model
        if self.backbone in ("hybrid", "transformer") and width % self.n_heads:
            raise ValueError("attention width must be divisible by n_heads")

    @property
    def variational(self):
        return self.model_name.endswith("_vae")

    @property
    def backbone(self):
        return self.model_name.rsplit("_", 1)[0]

    @property
    def receptive_field(self):
        return 1 + 2 * (self.tcn_kernel_size - 1) * (2**self.tcn_levels - 1)


@dataclass
class TrainConfig:
    epochs: int = 40
    lr: float = 1e-3
    weight_decay: float = 1e-4
    recon_loss: str = "mse"
    # Reconstruction: mean per input element. KL: SUM per latent vector.
    beta_target: float = .01
    kl_anneal_epochs: int = 5
    grad_clip_norm: float | None = 1.
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    selection_metric: str = "recon"  # or fixed_beta: recon + beta_target * KL
    teacher_forcing: bool = False
    device: str = "cpu"
    seed: int = 42
    deterministic: bool = False

    def __post_init__(self):
        if not isinstance(self.deterministic, bool):
            raise ValueError("deterministic must be boolean")
        positive_int("epochs", self.epochs)
        positive_int("early_stopping_patience", self.early_stopping_patience)
        if not isinstance(self.kl_anneal_epochs, int) or self.kl_anneal_epochs < 0:
            raise ValueError("kl_anneal_epochs must be nonnegative")
        for name in ("lr", "weight_decay", "beta_target", "early_stopping_min_delta"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0 or (name == "lr" and value == 0):
                raise ValueError(f"invalid {name}")
        if self.grad_clip_norm is not None and (not math.isfinite(self.grad_clip_norm) or self.grad_clip_norm <= 0):
            raise ValueError("grad_clip_norm must be positive or None")
        if self.recon_loss not in ("mse", "mae") or self.selection_metric not in ("recon", "fixed_beta"):
            raise ValueError("unsupported loss or selection metric")


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
