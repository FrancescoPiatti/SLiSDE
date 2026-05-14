# benchmarks/neural_sde_config.py
"""Configuration dataclass for the Neural SDE benchmark model.

Supports grid search via YAML configuration files, mirroring the SLiSDE
experiment workflow.
"""
from dataclasses import dataclass, field
from typing import Literal

DiffusionType = Literal["full", "time_only", "time_state"]
ActivationType = Literal["gelu", "tanh", "relu"]
# Normalization applied to the latent state ``z`` at the start of every
# Euler–Maruyama step.  "none" disables it (default, preserves legacy
# behaviour).
NormType = Literal["none", "layer", "rms"]


@dataclass
class NeuralSDEConfig:
    """Configuration for the Neural SDE benchmark.

    The Neural SDE uses MLP-based drift and diffusion:

        dZ = mu(Z, t) dt + sigma(Z, t) dW

    where mu is always an MLP of (Z, t/T), and sigma depends on
    ``diffusion_type``:

        full:       sigma(Z, t)  — MLP takes (Z, t/T)
        time_only:  sigma(t)     — MLP takes only t/T
        time_state: sigma(t)*Z   — MLP takes t/T, output multiplied by Z

    Attributes:
        dim:              latent state dimension.
        noise_dim:        Brownian motion dimension.
        output_dim:       output projection dimension.
        hidden_dim:       MLP hidden layer width.
        num_hidden_layers: number of hidden layers in drift and diffusion MLPs.
        diffusion_type:   diffusion variant.
        activation:       MLP activation function.
        n_steps:          number of Euler-Maruyama time steps.
        T:                terminal time.
        w_init_std:       MLP weight initialisation std.
        use_residual:     add residual connections in MLP (only when
                          num_hidden_layers >= 2 and hidden_dim == input dim).
        drift_scale:      multiplicative scaling on drift output (helps
                          stability for deeper MLPs).
        diffusion_scale:  multiplicative scaling on diffusion output.
    """

    # -- Dimensions --
    dim: int = 32
    noise_dim: int = 4
    output_dim: int = 1

    # -- MLP architecture --
    hidden_dim: int = 64
    num_hidden_layers: int = 2
    diffusion_type: DiffusionType = "full"

    # -- Activation --
    activation: ActivationType = "gelu"

    # -- Time discretisation --
    n_steps: int = 100
    T: float = 1.0

    # -- Initialisation --
    w_init_std: float = 0.1

    # -- Stability --
    use_residual: bool = False
    drift_scale: float = 1.0
    diffusion_scale: float = 1.0

    # -- Optional pre-step normalization of z --
    # When active, the MLP inputs that depend on z are normalized via
    # LayerNorm / RMSNorm at the start of each Euler–Maruyama step.  The
    # SDE state ``z`` itself is not normalized (that would change the
    # dynamics); only the MLP input is.
    norm_type: NormType = "none"

    def validate(self) -> None:
        """Raise ``ValueError`` if any field is inconsistent."""
        if self.dim < 1:
            raise ValueError("dim must be >= 1")
        if self.noise_dim < 1:
            raise ValueError("noise_dim must be >= 1")
        if self.output_dim < 1:
            raise ValueError("output_dim must be >= 1")
        if self.hidden_dim < 1:
            raise ValueError("hidden_dim must be >= 1")
        if self.num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be >= 1")
        if self.n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        if self.T <= 0:
            raise ValueError("T must be > 0")
        if self.diffusion_type not in ("full", "time_only", "time_state"):
            raise ValueError(f"Unknown diffusion_type: {self.diffusion_type}")
        if self.activation not in ("gelu", "tanh", "relu"):
            raise ValueError(f"Unknown activation: {self.activation}")
        if self.drift_scale <= 0:
            raise ValueError("drift_scale must be > 0")
        if self.diffusion_scale <= 0:
            raise ValueError("diffusion_scale must be > 0")
        if self.norm_type not in ("none", "layer", "rms"):
            raise ValueError(f"Unknown norm_type: {self.norm_type}")
