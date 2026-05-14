# v2/src/girsanov/__init__.py
"""
v2 Girsanov package.

Last-layer-only drift-tilt controllers (concat+linear variant from
``drift_tilt2.py``).  ``v2`` keeps **only** this variant — the legacy
Brownian-only / MLP-correction controllers from v1 are intentionally
dropped.
"""
from v3.src.girsanov.drift_tilt import (
    FFTTiltController,
    StateSpaceTiltController,
)
from v3.src.girsanov.drift_tilt_path import (
    PathConvolutionTiltController,
    BilinearPathTiltController,
)
from v3.src.girsanov.drift_tilt_utils import (
    compute_causal_convolution_tilt,
    build_controller_affine_flows,
    run_controller_parallel_scan,
    compute_rn_log_weight,
    tilt_brownian_increments,
)

__all__ = [
    "FFTTiltController",
    "StateSpaceTiltController",
    "PathConvolutionTiltController",
    "BilinearPathTiltController",
    "compute_causal_convolution_tilt",
    "build_controller_affine_flows",
    "run_controller_parallel_scan",
    "compute_rn_log_weight",
    "tilt_brownian_increments",
]
