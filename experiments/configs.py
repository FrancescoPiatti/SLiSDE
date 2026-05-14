# v2/experiments/configs.py
"""
Training configuration and experiment config grid builder for v3.

v2 ships the vanilla SLiSDE family plus an optional **last-layer-only
Girsanov tilt** (``drift_tilt2``-style: concat+linear with optional
``Z^{(L-1)}`` branch).  The grid builder filters every YAML key against
``SLiSDEConfig.__dataclass_fields__`` so legacy YAMLs silently drop
unsupported keys, and validation failures are skipped.

Defines:
    - TrainConfig: training hyperparameters
    - build_slisde_grid(): build SLiSDEConfig grid from YAML
    - build_neural_sde_grid(): build NeuralSDEConfig grid from YAML
    - model_identity_key(): canonical key for a model
"""
from dataclasses import dataclass
from itertools import product
from typing import Dict

import yaml

from v3.src.config import SLiSDEConfig
from v3.benchmarks.neural_sde_config import NeuralSDEConfig
from v3.benchmarks.slice_config import SLiCEConfig


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    num_epochs: int = 300
    batch_size: int = 512
    eval_batch_size: int = 2048
    seed: int = 42
    peak_lr: float = 1e-3
    warmup_steps: int = 100
    end_lr: float = 1e-5
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    log_every: int = 50
    eval_every: int = 100
    use_wandb: bool = False
    wandb_project: str = "slisde-calibration"
    n_seeds: int = 3
    clip_z: float = 5.0
    barrier_weight: float = 1.0
    barrier_batch_size: int = 512

    # ---- Girsanov-specific training knobs ----
    # Number of warm-up epochs run with ``force_no_girsanov=True``; only after
    # this many epochs does the trainer activate the last-layer tilt.  Set to
    # 0 to use Girsanov from the very first epoch.
    start_girsanov_after_epoch: int = 0
    # Number of additional Q-paths drawn alongside each P-batch during
    # Girsanov training.  Used by the dual-batch shared-prefix forward pass
    # to estimate barrier / OTM payoffs under Q with importance weights.
    girsanov_batch_size_q: int = 256

    # ---- SNIS / IS weight stabilisation ----
    # Symmetric clip on the negative log Radon-Nikodym log-weight ``-log_rn``
    # (i.e. on ``log w``) BEFORE exponentiating.  Caps the dynamic range of
    # the weights so a single outlier path can't dominate the SNIS sum.  Set
    # to 0 (default) to disable.  Practical values: 10 - 30 nats.
    log_weight_clip: float = 0.0
    # Hard upper-bound clip on the (max-shifted) weights ``w`` after
    # exponentiation but before normalisation.  Set to 0 (default) to
    # disable; >0 to cap the largest weight at this value.
    weight_clip: float = 0.0
    # Drop the optimiser update for this step when the loss or any gradient
    # entry is non-finite (NaN/Inf).  Cheap insurance against a single
    # poisoned Q-step bricking the rest of training.
    skip_nonfinite_updates: bool = True

    # ---- KL(Q‖P) regulariser on the Girsanov tilt ----
    # When > 0, add ``kl_weight * KL(Q‖P)`` to the Girsanov-step loss.
    # Under Girsanov, KL(Q‖P) = E_Q[log_rn] = E_Q[ Σ u·dW^P + ½ Σ |u|² dt ].
    # We estimate it from the same Q-paths used by the SNIS estimator
    # (cheap — ``log_rn_q`` is already returned by the model).  This penalises
    # tilts whose RN density is far from 1 in expectation, and is a common
    # remedy for SNIS instability / weight degeneracy on diag_gated_in_flow.
    # 0 (default) disables the penalty.  Practical values: 1e-3 .. 1e-1.
    kl_weight: float = 0.0
    # When ``kl_estimator='snis'`` the KL is computed as the SNIS estimate
    # under Q (E_Q[log_rn]).  When ``'p_quad'`` it is the cheap upper bound
    # ½ E_P[|u|² dt] — but this requires per-step ``u`` which the model does
    # not currently return, so we use the Q-side estimator by default.
    kl_estimator: str = "snis"


# ─────────────────────────────────────────────────────────────────────
# Model identity: defines what counts as the "same model"
# ─────────────────────────────────────────────────────────────────────

def model_identity_key(cfg, include_girsanov: bool = True) -> str:
    """Return canonical model identity string.

    Identity fields:
        - stacking_mode (residual / gated_in_flow / diag_gated_in_flow)
        - matrix_type (diag / blockdiag / dense)
        - dim
        - num_layers
        - noise_rho
        - time_dependent_vector_fields (when not 'none')
        - causal_conv flag (when enabled)
        - girsanov flag (when enabled and ``include_girsanov``)
    """
    if isinstance(cfg, NeuralSDEConfig):
        return (
            f"neural_sde_d{cfg.dim}_L{cfg.num_hidden_layers}"
            f"_{cfg.diffusion_type}"
        )
    if isinstance(cfg, SLiCEConfig):
        # Identity captures every architectural axis we sweep.
        bs_tag = (
            f"bs{cfg.block_size}dd"
            if cfg.diagonal_dense
            else f"bs{cfg.block_size}"
        )
        return (
            f"slice_h{cfg.hidden_dim}_L{cfg.num_layers}"
            f"_{bs_tag}_{cfg.transition_mode}_{cfg.path_mode}_{cfg.bm_input_type}"
        )

    parts = [
        cfg.stacking_mode,
        cfg.matrix_type,
        f"d{cfg.dim}",
        f"L{cfg.num_layers}",
        f"rho{cfg.noise_rho}",
    ]
    if cfg.time_dependent_vector_fields != "none":
        parts.append(cfg.time_dependent_vector_fields)
    if getattr(cfg, "use_causal_conv", False):
        k = getattr(cfg, "causal_conv_kernel_size", 4)
        parts.append(f"cconv{k}")
    # Reflect the diag-only time-feature stacking in the identity key so a
    # sweep with diag_use_time_features ∈ {False, True} produces distinct
    # model identities.  Only meaningful for diag_gated_in_flow.
    if (
        cfg.stacking_mode == "diag_gated_in_flow"
        and not getattr(cfg, "diag_use_time_features", True)
    ):
        parts.append("notau")
    if include_girsanov and getattr(cfg, "use_girsanov", False):
        gp = [f"gir_{cfg.tilt_type}"]
        if cfg.tilt_type == "fft":
            gp.append(f"k{cfg.kernel_size}")
        if cfg.girsanov_use_z_dependence:
            gp.append("z")
            if cfg.girsanov_z_project_dim > 0:
                gp.append(f"zp{cfg.girsanov_z_project_dim}")
        if cfg.context_dim > 0:
            gp.append(f"ctx{cfg.context_dim}")
        gp.append(f"rL{getattr(cfg, 'girsanov_last_layer_rho', 0.9)}")
        parts.append("-".join(gp))
    return "_".join(parts)


# ─────────────────────────────────────────────────────────────────────
# YAML grid builders
# ─────────────────────────────────────────────────────────────────────

def build_slisde_grid(
    yaml_path: str,
    section: str = "grid",
    *,
    last_layer_only: bool = False,  # kept for call-site compat; no-op in v2
) -> Dict[str, SLiSDEConfig]:
    """Build SLiSDEConfig grid from YAML section.

    Returns dict mapping descriptive name -> SLiSDEConfig.

    All YAML keys not present in ``SLiSDEConfig.__dataclass_fields__`` are
    silently dropped, so legacy YAML files (with gate_type,
    bottleneck_rank, etc.) still parse.  Combos that fail validation are
    skipped.

    Girsanov configs are accepted: any section that sets ``use_girsanov:
    [true]`` (and optionally sweeps ``tilt_type``, ``kernel_size``,
    ``context_dim``, ``girsanov_use_z_dependence``,
    ``girsanov_z_project_dim``) yields ``SLiSDEConfig`` entries with the
    last-layer tilt enabled.
    """
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f)

    grid = raw.get(section, {})
    defaults = raw.get("defaults", {})
    if not grid:
        return {}

    param_names = list(grid.keys())
    param_values = [grid[k] if isinstance(grid[k], list) else [grid[k]] for k in param_names]

    valid = set(SLiSDEConfig.__dataclass_fields__.keys())
    configs = {}
    seen_keys = set()

    for combo in product(*param_values):
        kwargs = dict(zip(param_names, combo))
        full = {**defaults, **kwargs}

        # Handle noise_sharing from noise_rho (preserves v1 semantics).
        rho = full.get("noise_rho", 0.9)
        if rho == 1.0:
            full["noise_sharing"] = "shared"
        elif "noise_sharing" not in full:
            full["noise_sharing"] = "correlated"

        # Legacy backward-compat: ``use_layer_norm: false`` → norm_type='none'.
        if "use_layer_norm" in full:
            if full["use_layer_norm"] is False:
                full["norm_type"] = "none"
            full.pop("use_layer_norm", None)

        # Skip irrelevant combos: residual mode never reads x_prev, so a
        # causal conv there is meaningless and rejected by validate().
        stacking = full.get("stacking_mode", "residual")
        if stacking == "residual" and full.get("use_causal_conv", False):
            continue

        # ``diag_use_time_features`` is only consumed by diag_gated_in_flow.
        # Normalise to a single value for other modes so the YAML's [false,true]
        # sweep doesn't silently double the grid for stacking modes that ignore it.
        if stacking != "diag_gated_in_flow":
            full["diag_use_time_features"] = True

        # Filter to valid v2 fields.
        filtered = {k: v for k, v in full.items() if k in valid}

        # Deduplicate identical filtered combos (a YAML may sweep a key
        # that v2 ignores, producing repeated configs).
        key = tuple(sorted(filtered.items()))
        if key in seen_keys:
            continue
        seen_keys.add(key)

        try:
            cfg = SLiSDEConfig(**filtered)
            cfg.validate()
        except (ValueError, TypeError):
            continue

        # Build a descriptive name from the v2-relevant axes.
        parts = [stacking, full.get("matrix_type", "blockdiag")]
        if full.get("time_dependent_vector_fields", "none") != "none":
            parts.append(full.get("time_dependent_vector_fields"))
        parts.extend([
            f"rho{rho}",
            f"d{full.get('dim', 64)}",
            f"L{full.get('num_layers', 1)}",
            f"nd{full.get('noise_dim', 16)}",
            f"p{full.get('parallel_steps', 1)}",
        ])
        if full.get("use_causal_conv", False):
            parts.append(f"cconv{full.get('causal_conv_kernel_size', 4)}")
        if (
            stacking == "diag_gated_in_flow"
            and not full.get("diag_use_time_features", True)
        ):
            parts.append("notau")
        if full.get("norm_type", "rms") != "rms":
            parts.append(f"norm_{full.get('norm_type')}")
        if full.get("use_girsanov", False):
            gp = [f"gir-{full.get('tilt_type', 'state_space')}"]
            if full.get("tilt_type", "state_space") == "fft":
                gp.append(f"k{full.get('kernel_size', 64)}")
            if full.get("girsanov_use_z_dependence", False):
                gp.append("z")
                zp = full.get("girsanov_z_project_dim", 0)
                if zp > 0:
                    gp.append(f"zp{zp}")
            if full.get("context_dim", 0) > 0:
                gp.append(f"ctx{full.get('context_dim')}")
            gp.append(f"rL{full.get('girsanov_last_layer_rho', 0.9)}")
            parts.append("".join(gp))
        name = "_".join(str(p) for p in parts)

        # Disambiguate name collisions.
        base = name
        i = 2
        while name in configs:
            name = f"{base}_v{i}"
            i += 1
        configs[name] = cfg

    return configs


def build_slice_grid(
    yaml_path: str, section: str = "slice"
) -> Dict[str, SLiCEConfig]:
    """Build SLiCEConfig grid from YAML section.

    Mirrors ``build_neural_sde_grid``.  Filters to fields known by
    ``SLiCEConfig.__dataclass_fields__`` so legacy YAMLs drop unsupported
    keys silently, and skips combos that fail validation.
    """
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f)

    grid = raw.get(section, {})
    if not grid:
        return {}

    param_names = list(grid.keys())
    param_values = [
        grid[k] if isinstance(grid[k], list) else [grid[k]] for k in param_names
    ]
    valid = set(SLiCEConfig.__dataclass_fields__.keys())

    configs = {}
    for combo in product(*param_values):
        kwargs = dict(zip(param_names, combo))
        filtered = {k: v for k, v in kwargs.items() if k in valid}
        try:
            cfg = SLiCEConfig(**filtered)
            cfg.validate()
        except (ValueError, TypeError):
            continue

        bs_tag = (
            f"bs{filtered.get('block_size', 1)}dd"
            if filtered.get("diagonal_dense", False)
            else f"bs{filtered.get('block_size', 1)}"
        )
        name = (
            f"slice_h{filtered.get('hidden_dim', 64)}"
            f"_L{filtered.get('num_layers', 4)}"
            f"_{bs_tag}"
            f"_{filtered.get('transition_mode', 'euler')}"
            f"_{filtered.get('path_mode', 'values')}"
            f"_{filtered.get('bm_input_type', 'path')}"
            f"_d{filtered.get('noise_dim', 4)}"
        )
        # Disambiguate name collisions deterministically.
        base = name
        i = 2
        while name in configs:
            name = f"{base}_v{i}"
            i += 1
        configs[name] = cfg

    return configs


def build_neural_sde_grid(yaml_path: str, section: str = "neural_sde") -> Dict[str, NeuralSDEConfig]:
    """Build NeuralSDEConfig grid from YAML section."""
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f)

    grid = raw.get(section, {})
    if not grid:
        return {}

    param_names = list(grid.keys())
    param_values = [grid[k] if isinstance(grid[k], list) else [grid[k]] for k in param_names]
    valid = set(NeuralSDEConfig.__dataclass_fields__.keys())

    configs = {}
    for combo in product(*param_values):
        kwargs = dict(zip(param_names, combo))
        filtered = {k: v for k, v in kwargs.items() if k in valid}
        try:
            cfg = NeuralSDEConfig(**filtered)
            cfg.validate()
        except (ValueError, TypeError):
            continue

        name = (
            f"neural_sde_{filtered.get('diffusion_type', 'full')}"
            f"_h{filtered.get('hidden_dim', 64)}"
            f"_L{filtered.get('num_hidden_layers', 2)}"
            f"_d{filtered.get('dim', 32)}"
        )
        configs[name] = cfg

    return configs
