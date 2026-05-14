"""SLiSDE v3 layer modules.

v3 keeps two stacking modes:

* ``residual`` — classical baseline (additive residual on a token-wise MLP);
* ``gated_in_flow`` (default) — bilinear-gated diagonal F-scale + bilinear
  residual offset on g. (Equivalent to v2's ``diag_gated_in_flow``.)

The base SDE :class:`SLiSDE` has an optional ``time_dependent_vector_fields``
flag that activates the structure-aware coefficient decoder previously
implemented as a separate ``SLiSDETime3`` class.
"""
from v3.src.layers.base import SLiSDE
from v3.src.layers.residual import ResidualLayer
from v3.src.layers.gated_in_flow import GatedSLiSDELayer

__all__ = ["SLiSDE", "ResidualLayer", "GatedSLiSDELayer"]
