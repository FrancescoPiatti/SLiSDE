# benchmarks/slice_config.py
"""
Configuration dataclass for the SLiCE benchmark model.

Port of  ``datasig-ac-uk/slices``  (NeurIPS 2025 Spotlight).  Mirrors
``StackedSLiCE`` from the upstream repo: an embedding layer feeds a stack
of residual ``SLiCELayer`` blocks, each containing a structured-linear
controlled differential-equation core (``SLiCE``) followed by a token
MLP, with a final output projection.

For the SLNSDE benchmark, the input to the stack is a sample of the
driving Brownian motion (``W_t = ∫dW``) — i.e. the same source of
randomness that drives our SLiSDE family — so the benchmark consumes
exactly the same noise budget per path as ``SLiSDEModel`` and
``NeuralSDEStack``.  The default ``path_mode='values'`` makes SLiCE
internally differentiate the input back to the increments, recovering
a Mamba-1-style elementwise diagonal SSM driven by ``dW`` when
``block_size=1``.
"""
from dataclasses import dataclass
from typing import Literal

# Norm types reused from the upstream SLiCE repo.
NormType = Literal["rmsnorm", "layernorm"]
PathMode = Literal["values", "increments"]
TransitionMode = Literal["euler", "matrix_exp"]
FFStyle = Literal["mlp", "single"]
FFActivation = Literal["gelu", "glu", "tanh"]
DropoutPos = Literal["residual", "output"]
BMInputType = Literal["path", "increments"]


@dataclass
class SLiCEConfig:
    """Configuration for the SLiCE benchmark.

    Hyperparameter names mirror the upstream ``StackedSLiCE`` constructor
    where possible.  Three SLNSDE-specific fields:

      - ``noise_dim``: dimension of the driving Brownian motion (= input dim
        to the stack).
      - ``output_dim``: dimension of the final linear projection.
      - ``bm_input_type``:  if ``"path"`` (default), feed cumulative
        Brownian motion ``W_t`` to the stack with ``path_mode='values'``
        (SLiCE differentiates internally → driving = dW).  If
        ``"increments"``, feed ``dW`` directly with
        ``path_mode='increments'``.  Mathematically equivalent under a
        learnt linear embedding, but the path form is the one used in
        the SLiCE paper for neural-CDE-style modelling.
    """

    # -- Stack architecture --
    num_layers: int = 4
    hidden_dim: int = 64

    # -- I/O --
    noise_dim: int = 4   # = input_dim of the stack (Brownian motion dim)
    output_dim: int = 1

    # -- SLiCE core --
    block_size: int = 1                  # 1 ⇒ Mamba-1-style elementwise
    diagonal_dense: bool = False
    bias: bool = True
    init_std: float = 0.01
    scale: float = 1.0
    path_mode: PathMode = "values"
    transition_mode: TransitionMode = "euler"

    # -- SLiCELayer (residual wrapper) --
    norm_type: NormType = "rmsnorm"
    prenorm: bool = True
    second_norm: bool = True
    ff_style: FFStyle = "mlp"
    ff_activation: FFActivation = "gelu"
    ff_mult: int = 4
    dropout_position: DropoutPos = "residual"
    norm_eps: float = 1e-6

    # -- BM driving --
    n_steps: int = 256
    T: float = 1.0
    bm_input_type: BMInputType = "path"

    def validate(self) -> None:
        if self.num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if self.hidden_dim < 1:
            raise ValueError("hidden_dim must be >= 1")
        if self.noise_dim < 1:
            raise ValueError("noise_dim must be >= 1")
        if self.output_dim < 1:
            raise ValueError("output_dim must be >= 1")
        if self.block_size < 1:
            raise ValueError("block_size must be >= 1")
        if not self.diagonal_dense and self.hidden_dim % self.block_size != 0:
            raise ValueError(
                "hidden_dim must be divisible by block_size when "
                "diagonal_dense=False"
            )
        if self.path_mode not in ("values", "increments"):
            raise ValueError("path_mode must be 'values' or 'increments'")
        if self.transition_mode not in ("euler", "matrix_exp"):
            raise ValueError("transition_mode must be 'euler' or 'matrix_exp'")
        if self.norm_type not in ("rmsnorm", "layernorm"):
            raise ValueError("norm_type must be 'rmsnorm' or 'layernorm'")
        if self.ff_style not in ("mlp", "single"):
            raise ValueError("ff_style must be 'mlp' or 'single'")
        if self.ff_activation not in ("gelu", "glu", "tanh"):
            raise ValueError("ff_activation must be 'gelu', 'glu', or 'tanh'")
        if self.ff_mult < 1:
            raise ValueError("ff_mult must be >= 1")
        if self.ff_style == "single" and self.ff_mult != 1:
            raise ValueError("ff_mult must be 1 when ff_style='single'")
        if self.dropout_position not in ("residual", "output"):
            raise ValueError(
                "dropout_position must be 'residual' or 'output'"
            )
        if self.n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        if self.T <= 0:
            raise ValueError("T must be > 0")
        if self.bm_input_type not in ("path", "increments"):
            raise ValueError("bm_input_type must be 'path' or 'increments'")
