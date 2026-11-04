"""Conditional paired intervals with explicit limits on temporal identification."""
import math
import numpy as np


def paired_block_interval(delta, block_len, repeats=2000, seed=2026, min_blocks=5):
    """Resample aligned time blocks jointly across initialization seeds.

    Five blocks is a conservative reporting policy, not a coverage guarantee.
    Invalid block lengths never silently turn a temporal CI into a seed-only CI.
    """
    delta = np.asarray(delta, dtype=float)
    if delta.ndim != 2 or not delta.size or not np.isfinite(delta).all():
        raise ValueError("delta must be a finite nonempty (seeds, time) array")
    if isinstance(block_len, bool) or not isinstance(block_len, int) or block_len < 1:
        raise ValueError("block_len must be a positive integer")
    if not isinstance(repeats, int) or repeats < 100 or min_blocks < 2:
        raise ValueError("Need at least 100 resamples and min_blocks >= 2")
    n_seeds, n = delta.shape
    result = dict(mean=float(delta.mean()), lower=None, upper=None, n_windows=n, n_seeds=n_seeds,
        block_len=block_len, approximate_blocks=n/block_len, resamples=repeats, min_blocks=min_blocks,
        status="insufficient_temporal_blocks", reason=None)
    if block_len >= n or n/block_len < min_blocks:
        result["reason"] = "Requested block leaves too few independent time blocks; no temporal interval reported"
        return result
    rng = np.random.default_rng(seed)
    estimates = np.empty(repeats)
    for i in range(repeats):
        selected = rng.integers(0, n_seeds, n_seeds)
        starts = rng.integers(0, n-block_len+1, math.ceil(n/block_len))
        indices = np.concatenate([np.arange(start, start+block_len) for start in starts])[:n]
        estimates[i] = delta[selected][:, indices].mean()
    result.update(lower=float(np.quantile(estimates, .025)), upper=float(np.quantile(estimates, .975)),
                  status="ok", reason="Conditional interval; block adequacy and coverage are not guaranteed")
    return result
