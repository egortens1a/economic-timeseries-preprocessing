"""Настройки данных, моделей и экспериментов с проверкой параметров."""
from dataclasses import dataclass, field
import math
from tsprep.data_preprocessing.denoising import FAMILIES

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
    training_target: str = "observed"
    validation_target: str = "observed"

    def __post_init__(self):
        if not isinstance(self.deterministic, bool):
            raise ValueError("deterministic must be boolean")
        if self.training_target not in ("observed", "clean") or self.validation_target not in ("observed", "clean"):
            raise ValueError("training_target and validation_target must be observed or clean")
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
        if self.recon_loss not in ("mse", "mae") or self.selection_metric not in ("recon", "fixed_beta", "endpoint_mae"):
            raise ValueError("unsupported loss or selection metric")


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


@dataclass(frozen=True)
class ImputationProtocol:
    """Параметры заполнения пропусков; значения по умолчанию для NN5."""
    window: int = 56
    period: int = 7
    train_stop: int = 511
    val_stop: int = 623
    epochs: int = 60
    batch_size: int = 256
    learning_rates: tuple = (.0003, .001)
    initialization_seeds: tuple = (111, 222, 333)
    validation_mask_seeds: tuple = (31001, 31002, 31003)
    test_mask_seeds: tuple = (41001, 41002, 41003)
    missing_fraction: float = .05
    pca_ranks: tuple = (4, 8, 16, 24)
    ridge_alphas: tuple = (.1, 1., 10., 100.)
    shrinkages: tuple = (0., .25, .5, 1.)
    bootstrap_repeats: int = 5000
    bootstrap_seed: int = 55013
    bootstrap_block: int = 4
    device: str = "cpu"
    threads: int = 1

    def validate(self, length):
        for name in ("window", "period", "train_stop", "val_stop", "epochs", "batch_size",
                     "bootstrap_repeats", "bootstrap_block", "threads"):
            positive_int(name, getattr(self, name))
        if self.window % self.period or self.window < self.period:
            raise ValueError("window must be a multiple of period")
        if not self.window < self.train_stop < self.val_stop < length:
            raise ValueError("require window < train_stop < val_stop < input length")
        if (self.val_stop-self.train_stop) % self.period or (length-self.val_stop) % self.period:
            raise ValueError("validation and test lengths must contain whole seasonal periods")
        if not math.isfinite(self.missing_fraction) or not 0 < self.missing_fraction < 1:
            raise ValueError("missing_fraction must be between zero and one")
        for name in ("initialization_seeds", "validation_mask_seeds", "test_mask_seeds"):
            group = getattr(self, name)
            if not group or len(group) != len(set(group)) or any(type(s) is not int or s < 0 for s in group):
                raise ValueError(f"{name} must contain unique nonnegative integers")
        if set(self.validation_mask_seeds) & set(self.test_mask_seeds):
            raise ValueError("validation and test masks require separate seeds")
        for name in ("learning_rates", "ridge_alphas"):
            if not getattr(self, name) or any(not math.isfinite(x) or x <= 0 for x in getattr(self, name)):
                raise ValueError(f"invalid {name}")
        if not self.pca_ranks or any(type(x) is not int or x < 1 for x in self.pca_ranks):
            raise ValueError("pca_ranks must contain positive integers")
        if not self.shrinkages or any(not math.isfinite(x) or not 0 <= x <= 1 for x in self.shrinkages):
            raise ValueError("shrinkages must lie in [0, 1]")
        if type(self.bootstrap_seed) is not int or self.bootstrap_seed < 0:
            raise ValueError("bootstrap_seed must be nonnegative")


@dataclass
class RepresentationProtocol:
    """Параметры сравнения эмбеддингов обычного AE и AE с удалением шума."""
    protocol_id: str = "embedding_robustness_v1"
    length: int = 6000
    window: int = 48
    latent_dim: int = 8
    hidden_dim: int = 16
    levels: int = 3
    training_stride: int = 4
    train_fraction: float = .7
    batch_size: int = 128
    epochs: int = 120
    learning_rates: tuple = (.0003, .001)
    development_seeds: tuple = (16001, 16002, 16003)
    confirmation_seeds: tuple = tuple(range(96001, 96021))
    initialization_seeds: tuple = (111, 222, 333)
    mad_thresholds: tuple = (3., 5., 7., 9.)
    forecast_horizon: int = 7
    bootstrap_repeats: int = 5000
    bootstrap_seed: int = 61017
    device: str = "cpu"
    smoke: bool = False
    families: tuple = FAMILIES

    def validate(self):
        from tsprep.data_preprocessing.denoising import FAMILIES
        for name in ("length", "window", "latent_dim", "hidden_dim", "levels", "training_stride",
                     "batch_size", "epochs", "forecast_horizon", "bootstrap_repeats"):
            positive_int(name, getattr(self, name))
        groups = (self.development_seeds, self.confirmation_seeds, self.initialization_seeds)
        if any(not s or len(s) != len(set(s)) for s in groups) or 111 not in self.initialization_seeds:
            raise ValueError("unique nonempty seed groups including initialization seed 111 required")
        if set(self.development_seeds) & set(self.confirmation_seeds) or len(self.confirmation_seeds) < 2:
            raise ValueError("need independent development and at least two confirmation seeds")
        if not self.families or len(set(self.families)) != len(self.families) or set(self.families)-set(FAMILIES):
            raise ValueError("invalid generator families")
        if not self.learning_rates or any(not math.isfinite(x) or x <= 0 for x in self.learning_rates):
            raise ValueError("invalid learning rates")
        if not self.mad_thresholds or any(not math.isfinite(x) or x <= 0 for x in self.mad_thresholds):
            raise ValueError("invalid MAD thresholds")
        if not 0 < self.train_fraction < 1 or self.latent_dim >= self.window:
            raise ValueError("invalid split or compression dimension")
        if min(int((self.length-1)*self.train_fraction), int((self.length-1)*(1-self.train_fraction))) < self.window+self.forecast_horizon:
            raise ValueError("trajectory segments too short")
