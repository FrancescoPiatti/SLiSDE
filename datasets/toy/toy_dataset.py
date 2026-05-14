# experiments/datasets/toy_dataset.py
"""
Ground-truth nonlinear SDE simulation and calibration dataset generation.

The ground-truth process is a 3-component nonlinear SDE system with
state-dependent diffusion:

    dX1 = -X1 (X1^2 - 1) dt + sqrt(|X1| + 1) * cbrt(X1 + 5) dW1     X1(0) = 0.5
    dX2 = sin(X2) (2 - X2^2) dt + (1 + X2^2)^{1/3} dW2               X2(0) = 0.0
    dX3 = (X3 - X3^3/3) dt + exp(-X3^2/4) * X3 dW3                    X3(0) = 1.0

Observable:
    Y_t = 0.4 * X1_t + 0.35 * X2_t + 0.25 * X3_t

Target functionals (evaluated at maturities T_i):
    1.  Soft-max payoff:    E[max(Y_T - K, 0)]   for several strikes K
    2.  Running maximum:    E[max_{0<=t<=T} Y_t]
    3.  Squared path avg:   E[(1/T) int_0^T Y_t^2 dt]
    4.  Barrier probability: P(max_{0<=s<=T_i} Y_s >= H_j)  for barrier levels H_j
"""
from dataclasses import dataclass
from typing import NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr


# ─────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────

@dataclass
class GroundTruthParams:
    """Parameters of the 3-component nonlinear SDE system."""
    x1_0: float = 0.5
    x2_0: float = 0.0
    x3_0: float = 1.0
    weights: tuple = (0.4, 0.35, 0.25)


@dataclass
class CalibrationDataset:
    """Pre-computed ground-truth targets for calibration.

    Attributes:
        target_payoffs:        (n_strikes, n_maturities) soft-max payoffs.
        target_running_max:    (n_maturities,) running maximum expectations.
        target_squared_avg:    (n_maturities,) squared path average expectations.
        strikes:               (n_strikes,) strike levels.
        maturities:            (n_maturities,) maturity times.
        maturity_indices:      (n_maturities,) integer indices into model time grid.
        gt_params:             ground-truth SDE parameters used.
        n_mc_paths:            number of Monte Carlo paths used.
        y0:                    (output_dim,) ground-truth initial observable value.
        barriers:              (n_barriers,) barrier levels from running-max quantiles.
        target_barrier_probs:  (n_barriers, n_maturities) barrier hit probabilities.
    """
    target_payoffs: jnp.ndarray
    target_running_max: jnp.ndarray
    target_squared_avg: jnp.ndarray
    strikes: jnp.ndarray
    maturities: jnp.ndarray
    maturity_indices: jnp.ndarray
    gt_params: GroundTruthParams
    n_mc_paths: int
    y0: jnp.ndarray
    barriers: jnp.ndarray = None
    target_barrier_probs: jnp.ndarray = None


# ─────────────────────────────────────────────────────────────────────
# SDE drift and diffusion
# ─────────────────────────────────────────────────────────────────────

def _drift(x: jnp.ndarray) -> jnp.ndarray:
    """Drift vector for the 3-component system."""
    x1, x2, x3 = x[..., 0], x[..., 1], x[..., 2]
    mu1 = -x1 * (x1 ** 2 - 1.0)
    mu2 = jnp.sin(x2) * (2.0 - x2 ** 2)
    mu3 = x3 - x3 ** 3 / 3.0
    return jnp.stack([mu1, mu2, mu3], axis=-1)


def _diffusion(x: jnp.ndarray) -> jnp.ndarray:
    """Diagonal diffusion coefficients for the 3-component system."""
    x1, x2, x3 = x[..., 0], x[..., 1], x[..., 2]
    sig1 = jnp.sqrt(jnp.abs(x1) + 1.0) * jnp.cbrt(x1 + 5.0)
    sig2 = jnp.cbrt(1.0 + x2 ** 2)
    sig3 = jnp.exp(-x3 ** 2 / 4.0) * x3
    return jnp.stack([sig1, sig2, sig3], axis=-1)


# ─────────────────────────────────────────────────────────────────────
# Euler-Maruyama simulation via jax.lax.scan
# ─────────────────────────────────────────────────────────────────────

class _ScanCarry(NamedTuple):
    x: jnp.ndarray     # (n_paths, 3)


def _euler_step(carry: _ScanCarry, dW_t: jnp.ndarray, dt: float):
    """One Euler-Maruyama step."""
    x = carry.x
    mu = _drift(x)
    sigma = _diffusion(x)
    x_new = x + mu * dt + sigma * dW_t
    return _ScanCarry(x=x_new), x_new


def simulate_ground_truth(
    key: jnp.ndarray,
    gt_params: GroundTruthParams,
    n_paths: int,
    n_steps: int,
    T: float,
) -> jnp.ndarray:
    """Simulate the 3-component SDE and return the observable Y_t.

    Returns:
        Y: (n_paths, n_steps + 1) observable paths, including initial value.
    """
    dt = T / n_steps
    sqrt_dt = jnp.sqrt(dt)
    w = jnp.array(gt_params.weights)

    x0 = jnp.broadcast_to(
        jnp.array([gt_params.x1_0, gt_params.x2_0, gt_params.x3_0]),
        (n_paths, 3),
    )

    dW = jr.normal(key, (n_steps, n_paths, 3)) * sqrt_dt

    def step_fn(carry, dW_t):
        new_carry, x_new = _euler_step(carry, dW_t, dt)
        return new_carry, x_new

    init_carry = _ScanCarry(x=x0)
    _, x_traj = jax.lax.scan(step_fn, init_carry, dW)

    x_full = jnp.concatenate([x0[None], x_traj], axis=0)
    Y = jnp.einsum("tpi,i->tp", x_full, w)
    return Y.T


# ─────────────────────────────────────────────────────────────────────
# Target functional computation
# ─────────────────────────────────────────────────────────────────────

def compute_targets(
    Y: jnp.ndarray,
    strikes: jnp.ndarray,
    maturity_indices: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute target functionals from simulated paths.

    Returns:
        target_payoffs:      (n_strikes, n_mats) soft-max payoffs.
        target_running_max:  (n_mats,) running maximum expectations.
        target_squared_avg:  (n_mats,) squared path averages.
    """
    # 1. Soft-max payoffs
    Y_at_mats = Y[:, maturity_indices]
    payoffs = jax.nn.relu(Y_at_mats[:, None, :] - strikes[None, :, None])
    target_payoffs = jnp.mean(payoffs, axis=0)

    # 2. Running maximum
    cummax = jax.lax.associative_scan(jnp.maximum, Y, axis=1)
    target_running_max = jnp.mean(cummax[:, maturity_indices], axis=0)

    # 3. Squared path average
    cumsq = jnp.cumsum(Y ** 2, axis=1)
    divisors = jnp.arange(1, Y.shape[1] + 1)[None, :]
    avg_sq = cumsq / divisors
    target_squared_avg = jnp.mean(avg_sq[:, maturity_indices], axis=0)

    return target_payoffs, target_running_max, target_squared_avg


def compute_barrier_targets(
    Y: jnp.ndarray,
    maturity_indices: jnp.ndarray,
    barrier_quantiles: Tuple[float, ...] = (0.80, 0.90, 0.95),
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute barrier hit probabilities from simulated paths.

    Barrier levels are chosen as empirical quantiles of the terminal
    running maximum distribution, then reused across all maturities.

    Args:
        Y:                  (n_paths, n_steps+1) observable paths.
        maturity_indices:   (n_maturities,) time indices.
        barrier_quantiles:  quantiles of the terminal running max to use as barriers.

    Returns:
        barriers:            (n_barriers,) barrier levels.
        target_barrier_probs: (n_barriers, n_maturities) barrier hit probabilities.
    """
    # Running maximum over time
    cummax = jax.lax.associative_scan(jnp.maximum, Y, axis=1)

    # Terminal running max for barrier level selection
    terminal_runmax = cummax[:, -1]
    barriers = jnp.array([
        jnp.percentile(terminal_runmax, q * 100.0) for q in barrier_quantiles
    ])

    # Running max at each maturity: (n_paths, n_maturities)
    runmax_at_mats = cummax[:, maturity_indices]

    # Barrier hit indicators: (n_paths, n_barriers, n_maturities)
    hit = (runmax_at_mats[:, None, :] >= barriers[None, :, None]).astype(jnp.float32)

    # Barrier probabilities: (n_barriers, n_maturities)
    target_barrier_probs = jnp.mean(hit, axis=0)

    return barriers, target_barrier_probs


# ─────────────────────────────────────────────────────────────────────
# Dataset generation
# ─────────────────────────────────────────────────────────────────────

def generate_dataset(
    gt_params: Optional[GroundTruthParams] = None,
    strikes: Optional[jnp.ndarray] = None,
    maturities: Optional[jnp.ndarray] = None,
    n_mc_paths: int = 200_000,
    model_n_steps: int = 100,
    model_T: float = 1.0,
    key: jnp.ndarray = jr.PRNGKey(0),
    barrier_quantiles: Tuple[float, ...] = (0.80, 0.90, 0.95),
) -> CalibrationDataset:
    """Generate the calibration dataset.

    Simulates the ground-truth nonlinear SDE system via Monte Carlo,
    then computes target functionals at specified maturities and strikes,
    including barrier hit probabilities.
    """
    if gt_params is None:
        gt_params = GroundTruthParams()
    if strikes is None:
        strikes = jnp.array([-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
    if maturities is None:
        maturities = jnp.array([0.1, 0.25, 0.5, 0.75, 1.0])

    dt = model_T / model_n_steps
    maturity_indices = jnp.round(maturities / dt).astype(jnp.int32)
    maturity_indices = jnp.clip(maturity_indices, 1, model_n_steps)

    print(f"Simulating {n_mc_paths:,} ground-truth paths "
          f"({model_n_steps} steps, T={model_T})...")
    Y = simulate_ground_truth(key, gt_params, n_mc_paths, model_n_steps, model_T)

    print("Computing target functionals...")
    target_payoffs, target_running_max, target_squared_avg = compute_targets(
        Y, strikes, maturity_indices,
    )

    print("Computing barrier targets...")
    barriers, target_barrier_probs = compute_barrier_targets(
        Y, maturity_indices, barrier_quantiles,
    )

    w = jnp.array(gt_params.weights)
    x0_vec = jnp.array([gt_params.x1_0, gt_params.x2_0, gt_params.x3_0])
    y0 = jnp.array([jnp.dot(w, x0_vec)])

    n_targets = strikes.shape[0] * maturities.shape[0] + 2 * maturities.shape[0]
    n_barrier_targets = barriers.shape[0] * maturities.shape[0]
    print(f"Dataset ready: {strikes.shape[0]} strikes x {maturities.shape[0]} maturities "
          f"= {n_targets} standard targets + {n_barrier_targets} barrier targets")
    print(f"  Barrier levels: {barriers}")
    print(f"  Barrier probs shape: {target_barrier_probs.shape}")

    return CalibrationDataset(
        target_payoffs=target_payoffs,
        target_running_max=target_running_max,
        target_squared_avg=target_squared_avg,
        strikes=strikes,
        maturities=maturities,
        maturity_indices=maturity_indices,
        gt_params=gt_params,
        n_mc_paths=n_mc_paths,
        y0=y0,
        barriers=barriers,
        target_barrier_probs=target_barrier_probs,
    )
