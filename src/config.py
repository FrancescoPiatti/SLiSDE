# v3/src/config.py
"""Unified configuration for the v3 SLiSDE framework.

v3 keeps two stacking modes:

* ``residual``       — classical residual baseline
* ``gated_in_flow``  (default) — bilinear-gated diagonal F-scale + bilinear
  residual offset on g (this is what v2 used to call ``diag_gated_in_flow``;
  the v2 ``gated_in_flow`` full-row gate is dropped).

Time dependence of the structured coefficients is controlled by a single
boolean :attr:`time_dependent_vector_fields` (replaces v2's
``time_gate_variant ∈ {"none", "base_time3"}``). When True, every
structured coefficient is augmented with a small zero-initialised decoder
on the deterministic time features ``τ(t) = (1, t, t², sin ωt, cos ωt)``.

The optional last-layer Girsanov tilt is unchanged from v2.
"""
from dataclasses import dataclass
from typing import Literal

MatrixType = Literal["diag", "blockdiag", "dense"]
NoiseType = Literal["multiplicative", "additive", "general"]
NoiseSharing = Literal["shared", "independent", "correlated"]
ActivationType = Literal["tanh", "gelu", "glu"]
StackingMode = Literal["residual", "gated_in_flow"]
NormType = Literal["none", "layer", "rms"]
TiltType = Literal["fft", "state_space", "path_conv", "bilinear_path"]


@dataclass
class SLiSDEConfig:
    """Configuration for the v3 Structured Linear Neural SDE stack.

    The SDE takes the form::

        dZ = (A_t Z + b_t) dt + diffusion_t(Z, dW)

    with diffusion controlled by ``noise_type``:

    * ``multiplicative``: ``Σ_k C^k_t Z dW^k``
    * ``additive``:       ``Σ_k D^k_t dW^k``
    * ``general``:        ``Σ_k (C^k_t Z + D^k_t) dW^k``

    All structured coefficients become time-dependent when
    :attr:`time_dependent_vector_fields` is ``True``.
    """

    # -- Dimensions / structure --
    dim: int = 64
    noise_dim: int = 16
    matrix_type: MatrixType = "blockdiag"
    block_size: int = 8
    noise_type: NoiseType = "multiplicative"

    # -- Time discretisation --
    n_steps: int = 2048
    T: float = 1.0

    # -- Time-dependent structured coefficients --
    # When True, each of (A, b, C^j, D) is augmented with a zero-init
    # decoder on the deterministic time features τ(t) so the layer
    # becomes time-inhomogeneous. False ⇒ the v2 static SLiSDE behaviour.
    time_dependent_vector_fields: bool = False
    time_embed_dim: int = 8
    time_omega: float = 6.283185307179586  # 2π

    # -- Initialisation --
    w_init_std: float = 0.25
    approximate_exponential: bool = True

    # -- Additive diffusion D parameterisation (LoRA) --
    # lora_rank > 0: D = D_A @ D_B with D_A: (noise_dim, lora_rank),
    #                 D_B: (lora_rank, dim).
    # lora_rank == 0: D is a full dense (noise_dim, dim) matrix.
    lora_rank: int = 0

    # -- Stacking --
    num_layers: int = 1
    stacking_mode: StackingMode = "gated_in_flow"

    # -- Gated-in-flow knob (caps |α − 1| ≤ ε on the diagonal F-scale) --
    final_scale: float = 0.1

    # -- Activation (used by residual stacking only) --
    activation: ActivationType = "tanh"

    # -- Noise sharing across layers --
    noise_sharing: NoiseSharing = "correlated"
    noise_rho: float = 0.9

    # -- Parallelism --
    parallel_steps: int = 1

    # -- Normalisation (always required by gated_in_flow) --
    norm_type: NormType = "rms"

    # -- Optional causal 1-D conv on x_prev between stacked layers --
    use_causal_conv: bool = False
    causal_conv_kernel_size: int = 4

    # -- Gated-in-flow time-feature augmentation --
    # When True, the gated layer concatenates (t, t², sin ωt, cos ωt)
    # to the gate input *after* the optional causal conv and *after* the
    # norm. Ignored when stacking_mode='residual'.
    diag_use_time_features: bool = True

    # -- Last-layer-only Girsanov tilt --
    use_girsanov: bool = False
    tilt_type: TiltType = "state_space"
    kernel_size: int = 64                # FFT controller kernel size
    context_dim: int = 0                 # optional xi-based additive shift
    girsanov_use_z_dependence: bool = False
    girsanov_z_project_dim: int = 0
    girsanov_state_expansion: int = 1
    girsanov_use_shift: bool = True

    # Correlation between the prefix and last-layer Brownian increments.
    # ρ=0 ⇒ fully independent last-layer Brownian (max tilt freedom);
    # ρ=1 ⇒ shared Brownian (no tilt freedom, equivalent to use_girsanov=False).
    girsanov_last_layer_rho: float = 0.9

    # Number of independent last-layer tilt controllers (used by exp_dax to
    # assign one to OTM-put SNIS and one to OTM-call SNIS).
    num_girsanov: int = 1

    # -- Output --
    output_dim: int = 1

    # -- Brownian-motion time augmentation --
    augment_brownian_with_time: bool = False

    # -- Learned prior over z0 --
    learn_z0_prior: bool = False

    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Run all dataclass-level sanity checks."""
        if self.dim < 1:
            raise ValueError("dim must be >= 1")
        if self.noise_dim < 1:
            raise ValueError("noise_dim must be >= 1")
        if self.block_size < 1:
            raise ValueError("block_size must be >= 1")
        if self.w_init_std <= 0:
            raise ValueError("w_init_std must be > 0")
        if self.n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        if self.T <= 0:
            raise ValueError("T must be > 0")
        if self.num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if self.output_dim < 1:
            raise ValueError("output_dim must be >= 1")
        if self.parallel_steps < 1:
            raise ValueError("parallel_steps must be >= 1")
        if self.time_embed_dim < 1:
            raise ValueError("time_embed_dim must be >= 1")
        if self.time_omega <= 0:
            raise ValueError("time_omega must be > 0")
        if self.norm_type not in ("none", "layer", "rms"):
            raise ValueError(f"Unknown norm_type: {self.norm_type}")
        if not (0.0 <= self.noise_rho <= 1.0):
            raise ValueError("noise_rho must be in [0, 1]")
        if self.lora_rank < 0:
            raise ValueError("lora_rank must be >= 0")
        if self.matrix_type == "blockdiag" and self.dim % self.block_size != 0:
            raise ValueError("For blockdiag, dim must be divisible by block_size.")
        if self.stacking_mode not in ("residual", "gated_in_flow"):
            raise ValueError(
                f"v3 only supports stacking_mode ∈ {{'residual','gated_in_flow'}} "
                f"(got {self.stacking_mode!r})"
            )
        if self.stacking_mode == "gated_in_flow" and self.norm_type not in ("layer", "rms"):
            raise ValueError(
                "stacking_mode='gated_in_flow' requires norm_type ∈ {'layer','rms'} "
                "(x_prev is always normalised)"
            )
        if self.final_scale <= 0:
            raise ValueError("final_scale must be > 0")
        if self.use_causal_conv:
            if self.stacking_mode == "residual":
                raise ValueError(
                    "use_causal_conv=True is only meaningful for gated_in_flow "
                    "(residual stacking does not consume x_prev)."
                )
            if self.causal_conv_kernel_size < 1:
                raise ValueError("causal_conv_kernel_size must be >= 1")
        if self.num_girsanov < 1:
            raise ValueError("num_girsanov must be >= 1")
        if self.num_girsanov > 1 and not self.use_girsanov:
            raise ValueError("num_girsanov > 1 requires use_girsanov=True")
        if self.use_girsanov:
            if not (0.0 <= self.girsanov_last_layer_rho <= 1.0):
                raise ValueError("girsanov_last_layer_rho must be in [0, 1]")
            if self.tilt_type not in ("fft", "state_space", "path_conv", "bilinear_path"):
                raise ValueError(f"Unknown tilt_type: {self.tilt_type!r}")
            if (
                self.tilt_type in ("fft", "path_conv", "bilinear_path")
                and self.kernel_size < 1
            ):
                raise ValueError("kernel_size must be >= 1 for FFT / path-conv tilts")
            if self.context_dim < 0:
                raise ValueError("context_dim must be >= 0")
            if self.girsanov_use_z_dependence and self.num_layers < 2:
                raise ValueError(
                    "girsanov_use_z_dependence requires num_layers >= 2"
                )
            if self.girsanov_z_project_dim < 0:
                raise ValueError("girsanov_z_project_dim must be >= 0")
            if self.tilt_type == "bilinear_path":
                if not self.girsanov_use_z_dependence:
                    raise ValueError(
                        "tilt_type='bilinear_path' requires girsanov_use_z_dependence=True"
                    )
                if self.num_layers < 2:
                    raise ValueError(
                        "tilt_type='bilinear_path' requires num_layers >= 2"
                    )
