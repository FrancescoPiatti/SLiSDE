# v2/src/__init__.py
"""
SLiSDE v2 — trimmed Structured Linear Neural SDE framework.

v2 keeps only the three stacking-mode layers (residual, gated_in_flow,
diag_gated_in_flow), the base SDE with optional ``base_time3`` time-gate
variant, and a single ``SLiSDEModel`` (no Girsanov, no last-layer-only
variant).  An optional causal 1D convolution on ``x_prev`` (between
stacked layers) is provided as a config flag.

Public API:
  SLiSDEConfig   — unified configuration dataclass
  SLiSDEModel    — unified model class
  NoiseGenerator — Brownian increment sampling
"""
from v3.src.config import SLiSDEConfig
from v3.src.models.model import SLiSDEModel
from v3.src.noise_gen import NoiseGenerator

__all__ = ["SLiSDEConfig", "SLiSDEModel", "NoiseGenerator"]
