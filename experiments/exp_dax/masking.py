# experiments/exp_dax2/masking.py
"""
Train/eval strike masking for overfitting detection.

Randomly selects a fraction of strikes as held-out evaluation targets.
The mask is deterministic given the seed, so it's reproducible across runs.
"""
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


@dataclass
class StrikeMasks:
    """Boolean masks over strike indices.  True = used for training."""
    call_train: jnp.ndarray       # (n_call_strikes,)
    put_train: jnp.ndarray        # (n_put_strikes,)
    otm_put_train: jnp.ndarray    # (n_otm_strikes,)
    otm_call_train: jnp.ndarray   # (n_otm_call_strikes,)
    digital_train: jnp.ndarray    # (n_digital_strikes,)

    @property
    def call_eval(self):
        return ~self.call_train

    @property
    def put_eval(self):
        return ~self.put_train

    @property
    def otm_put_eval(self):
        return ~self.otm_put_train

    @property
    def otm_call_eval(self):
        return ~self.otm_call_train

    @property
    def digital_eval(self):
        return ~self.digital_train

    def summary(self):
        parts = [
            ("calls", self.call_train),
            ("puts", self.put_train),
            ("otm_puts", self.otm_put_train),
            ("otm_calls", self.otm_call_train),
            ("digitals", self.digital_train),
        ]
        lines = []
        total_train, total_eval = 0, 0
        for name, mask in parts:
            n_train = int(mask.sum())
            n_total = len(mask)
            n_eval = n_total - n_train
            total_train += n_train
            total_eval += n_eval
            lines.append(f"  {name}: {n_train} train / {n_eval} eval (of {n_total})")
        lines.insert(0, f"Strike masks: {total_train} train, {total_eval} eval")
        return "\n".join(lines)


def create_strike_masks(dataset, eval_frac: float = 0.25, seed: int = 0) -> StrikeMasks:
    """Create random train/eval split masks for all strike types.

    Each strike type is split independently. At least 1 strike per type
    is always held out for eval, and at least 2 are kept for training.

    Args:
        dataset: DaxOptionsDataset with strike arrays.
        eval_frac: fraction of strikes to hold out (default 0.25).
        seed: random seed for reproducibility.

    Returns:
        StrikeMasks with boolean arrays (True = train).
    """
    rng = np.random.RandomState(seed)

    def _make_mask(n_strikes):
        n_eval = max(1, int(round(n_strikes * eval_frac)))
        n_eval = min(n_eval, n_strikes - 2)  # keep at least 2 for training
        indices = rng.permutation(n_strikes)
        eval_indices = set(indices[:n_eval].tolist())
        mask = jnp.array([i not in eval_indices for i in range(n_strikes)])
        return mask

    return StrikeMasks(
        call_train=_make_mask(len(dataset.call_strikes)),
        put_train=_make_mask(len(dataset.put_strikes)),
        otm_put_train=_make_mask(len(dataset.otm_put_strikes)),
        otm_call_train=_make_mask(len(dataset.otm_call_strikes)),
        digital_train=_make_mask(len(dataset.digital_put_strikes)),
    )
