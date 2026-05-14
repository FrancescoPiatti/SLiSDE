# datasets/dax/dax2_dataset.py
"""
Dense DAX options dataset for overfitting detection (exp_opt2).

Same structure as dax_dataset but with much denser strike grids,
designed to be split into train/eval subsets.
"""
from typing import Optional, List, Tuple

from v3.datasets.dax.dax_dataset import DaxOptionsDataset, load_dax_options


# Dense defaults: ~485 targets vs ~220 in dax1
DENSE_DEFAULTS = dict(
    n_call_strikes=25,
    n_put_strikes=25,
    n_otm_strikes=20,
    n_otm_call_strikes=15,
    n_digital_strikes=12,
    call_k_range=(0.88, 1.12),
    put_k_range=(0.88, 1.12),
    otm_k_range=(0.60, 0.88),
    otm_call_k_range=(1.08, 1.35),
    digital_k_range=(0.65, 0.98),
)


def load_dax2_options(
    csv_path: Optional[str] = None,
    spot: float = 25280.0,
    model_T: float = 1.0,
    model_n_steps: int = 256,
    **overrides,
) -> DaxOptionsDataset:
    """Load a dense DAX options dataset for train/eval splitting.

    Uses the same loader as dax1 but with denser default strike grids.
    Any kwarg from load_dax_options can be overridden.
    """
    kwargs = {**DENSE_DEFAULTS, "csv_path": csv_path, "spot": spot,
              "model_T": model_T, "model_n_steps": model_n_steps}
    kwargs.update(overrides)
    return load_dax_options(**kwargs)
