"""Параметры эксперимента по исправлению ошибок наблюдения."""
from dataclasses import dataclass, field
import math
from tsprep.data_preprocessing.denoising import FAMILIES


@dataclass
class DenoisingProtocol:
    protocol_id: str = "residual_denoising_v2"
    length: int = 6000
    development_seeds: tuple = (12001, 12002, 12003)
    validation_seeds: tuple = ()
    volatility_scales: tuple = (1.,)
    validation_upper_quantile: float | None = None
    validation_bootstrap_repeats: int = 2000
    confirmation_seeds: tuple = tuple(range(92001, 92021))
    initialization_seeds: tuple = (111, 222, 333)
    window: int = 48
    training_stride: int = 4
    train_fraction: float = .7
    hidden_dim: int = 16
    levels: int = 3
    local_context: int = 0
    context_mode: str = "local"
    correction_delay: int = 0
    delay_mode: str = "next"
    epochs: int = 120
    batch_size: int = 512
    learning_rates: tuple = (.0003, .001)
    losses: tuple = ("mae", "mse")
    identity_weights: tuple = (.1, 1., 10.)
    guard_scope: str = "average"
    transfer_checks: bool = False
    large_move_quantile: float = .95
    validation_every: int = 5
    guard_limit: float = 1.05
    forecast_horizon: int = 7
    forecast_features: int = 16
    bootstrap_repeats: int = 5000
    bootstrap_seed: int = 61017
    device: str = "cpu"
    smoke: bool = False
    families: tuple = field(default=FAMILIES)

    def validate(self):
        if (not self.volatility_scales or len(set(self.volatility_scales)) != len(self.volatility_scales) or
            any(not math.isfinite(s) or s <= 0 for s in self.volatility_scales)):
            raise ValueError("invalid volatility scales")
        if len(self.development_seeds) % len(self.volatility_scales):
            raise ValueError("development seeds must balance volatility scales")
        if self.validation_seeds:
            if len(self.validation_seeds) != len(self.development_seeds) or len(set(self.validation_seeds)) != len(self.validation_seeds):
                raise ValueError("validation seeds must be unique and paired with development seeds")
            if set(self.validation_seeds) & (set(self.development_seeds) | set(self.confirmation_seeds)):
                raise ValueError("train, validation and confirmation seeds must be disjoint")
        if self.validation_upper_quantile is not None:
            if not .5 < self.validation_upper_quantile < 1 or self.guard_scope != "family":
                raise ValueError("validation confidence guard requires family scope and a valid quantile")
            if len(self.development_seeds)//len(self.volatility_scales) < 2:
                raise ValueError("confidence guard needs at least two trajectories per volatility stratum")
        if self.validation_bootstrap_repeats < 100:
            raise ValueError("too few validation bootstrap repeats")
        if not self.losses or len(set(self.losses)) != len(self.losses) or not set(self.losses) <= {"mae", "mse", "mse_identity"}:
            raise ValueError("invalid residual losses")
        if not self.identity_weights or len(set(self.identity_weights)) != len(self.identity_weights) or any(not math.isfinite(w) or w < 0 for w in self.identity_weights):
            raise ValueError("invalid identity weights")
        if self.guard_scope not in ("average", "family") or not 0 < self.large_move_quantile < 1:
            raise ValueError("invalid preservation criteria")
        if not self.families or len(set(self.families)) != len(self.families) or not set(self.families) <= set(FAMILIES):
            raise ValueError("invalid families")
        if not self.learning_rates or any(not math.isfinite(lr) or lr <= 0 for lr in self.learning_rates):
            raise ValueError("invalid learning rates")
        if set(self.development_seeds) & set(self.confirmation_seeds):
            raise ValueError("development and confirmation seeds must be disjoint")
        for values in (self.development_seeds, self.confirmation_seeds, self.initialization_seeds):
            if not values or len(values) != len(set(values)):
                raise ValueError("empty or duplicate seeds")
        if 111 not in self.initialization_seeds:
            raise ValueError("initialization seeds must include the tuning seed 111")
        if not 0 < self.train_fraction < 1 or self.guard_limit < 1:
            raise ValueError("invalid split or guard")
        if min(self.epochs, self.batch_size, self.training_stride, self.validation_every) < 1:
            raise ValueError("invalid training budget")
        if not isinstance(self.local_context, int) or self.local_context < 0:
            raise ValueError("invalid local context")
        if self.context_mode not in ("local", "repeat") or (self.context_mode == "repeat" and not self.local_context):
            raise ValueError("invalid context mode")
        if self.correction_delay not in (0, 1) or self.delay_mode not in ("past", "next"):
            raise ValueError("unsupported correction delay")
        if self.correction_delay and self.local_context:
            raise ValueError("delay experiment excludes local context")
        if not self.correction_delay and self.delay_mode != "next":
            raise ValueError("past control requires delay=1")
        receptive_field = 1 + 4 * (2**self.levels-1) + self.local_context + self.correction_delay
        if self.correction_delay:
            receptive_field = max(receptive_field, 18)  # pair baseline: 16 past + target + next
        if self.window-self.forecast_features+1 < receptive_field:
            raise ValueError("forecast features need a complete causal receptive field")
        if int((self.length-1)*(1-self.train_fraction)) < self.window+self.forecast_horizon:
            raise ValueError("validation segment too short")

