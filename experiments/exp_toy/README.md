# `exp_toy` — long-path path-functional calibration

Trains SLiSDE, Neural-SDE and SLiCE on the `toy` dataset
(`v3/datasets/toy`). All four supervised targets — payoffs, running
maximum, squared path average, barrier hit probabilities — are evaluated
together. The barrier loss is the only rare-event component and is the
target of the optional Girsanov tilt.

## Models

The trainer dispatches automatically on the config type:

| Config | Built model |
|--------|-------------|
| `SLiSDEConfig`  | `v3.src.models.model.SLiSDEModel` |
| `NeuralSDEConfig` | `v3.benchmarks.neural_sde.NeuralSDEStack` |
| `SLiCEConfig`   | `v3.benchmarks.slice_model.SLiCEStack` |

Only SLiSDE has a Girsanov branch; Neural-SDE and SLiCE are run with the
plain vanilla loss.

## Trainers

* [`trainer_vanilla.py`](trainer_vanilla.py) — `VanillaTrainer`
  ($\mathbb P$-only). Computes
  *payoffs / runmax / sqavg / barrier* MSE on a single batch.
* [`trainer_girsanov.py`](trainer_girsanov.py) — `GirsanovTrainer`. Same
  three vanilla components plus a SNIS estimator for the barrier loss
  under $\mathbb Q$, with the standard dual-batch shared-prefix forward.
  A single controller is sufficient (`num_girsanov=1`) — the only
  rare-event target is the barrier.

## Loss assembly

```
L = (L_payoffs + L_runmax + L_sqavg + barrier_weight * L_barrier)
    / (1 + barrier_weight)
```

`L_payoffs`, `L_runmax`, `L_sqavg` always estimated under $\mathbb P$.
`L_barrier` is estimated under $\mathbb P$ during warm-up and under
$\mathbb Q$ via SNIS after `start_girsanov_after_epoch`. The reported
metric `loss_mse` excludes the KL anchor so Girsanov-on vs. off runs are
on the same scale.

## Backbone / controller gradient split

`GirsanovTrainer` uses **two AdamW optimisers** filtered by NNX path
(`tilt_controller` matches the controller params, the complement matches
the backbone). This isolates the SNIS-noisy barrier gradient from the
backbone so the P-side losses cannot be corrupted. See the trainer
docstring for the rationale.

## Running

```bash
# Quick smoke test on one tiny config:
python -m v3.experiments.exp_toy.run_experiment --quick

# Full vanilla sweep + SLiCE baseline:
python -m v3.experiments.exp_toy.run_experiment --include-slice

# Girsanov sweep (assumes the vanilla sweep has already picked a backbone):
python -m v3.experiments.exp_toy.run_experiment --mode girsanov
```

Common flags:

* `--mode {vanilla,girsanov,both}` — which block(s) to run.
* `--n-seeds N` — number of seeds per config (summary reports mean and
  std across seeds; std is NaN when `N=1`).
* `--include-neural-sde` / `--include-slice` — enable the benchmark blocks.
* `--start-girsanov-after-epoch K` — warm-up epochs under $\mathbb P$.
* `--girsanov-batch-size-q B` — extra $\mathbb Q$-paths per dual-batch step.

## Outputs

For each block (`vanilla`, `girsanov`, `slice`, `neural_sde`) the run
script saves:

* `all_losses_<block>.csv` — one row per (config, seed) with all final metrics
* `summary_<block>.csv` — one row per *identity* (= per architectural
  signature); reports `avg_<metric>` and `std_<metric>` across seeds.
  `std_*` is `NaN` when only one seed was run.
* `best_hyperparams_<block>.{json,pkl}` — the best config per identity.
