# `exp_dax` — masked DAX option-surface calibration

Trains SLiSDE, Neural-SDE and SLiCE on the dense `dax` option surface
(`v3/datasets/dax`). Each strike group (calls, puts, OTM puts, OTM calls,
digital puts) has 25% of its strikes held out as the eval set; the
remaining strikes train. We report train MSE, eval MSE, the eval–train
gap, and per-group eval MAE.

## Models

The trainer dispatches automatically on the config type:

| Config | Built model |
|--------|-------------|
| `SLiSDEConfig`  | `v3.src.models.model.SLiSDEModel` |
| `NeuralSDEConfig` | `v3.benchmarks.neural_sde.NeuralSDEStack` |
| `SLiCEConfig`   | `v3.benchmarks.slice_model.SLiCEStack` |

Only SLiSDE has a Girsanov branch.

## Trainers

* [`trainer_vanilla.py`](trainer_vanilla.py) — `VanillaTrainer`
  ($\mathbb P$-only, masked MSE across the five strike groups).
* [`trainer_girsanov.py`](trainer_girsanov.py) — `GirsanovTrainer`.
  Always trains **two** Girsanov controllers (one for the OTM-put SNIS
  estimator, one for the OTM-call SNIS estimator). The trainer pins
  `cfg.num_girsanov = 2` defensively, and the YAML grid hard-codes the
  same value. Digital puts continue to be estimated under $\mathbb P$.

The vanilla forward and the SNIS-only forward both share the same prefix
on the joint batch; the last layer is run twice (once per controller)
with the OTM-side samples re-weighted by the matching log-RN.

## Loss assembly

```
L = (w_call*L_call + w_put*L_put + w_otm*L_otm + w_otm_call*L_otm_call + w_digital*L_digital)
    / (w_call + w_put + w_otm + w_otm_call + w_digital)
```

The masked MSE `_masked_mse(pred, target, mask)` averages squared errors
only over the active strikes (train or eval). The KL anchor is added
only to the gradient target; the reported `loss_mse` (and per-group
`loss_call`, …) exclude it so Girsanov-on vs. off comparisons stay on
the same scale.

## Running

```bash
# Quick smoke test:
python -m v3.experiments.exp_dax.run_experiment --quick

# Full vanilla sweep:
python -m v3.experiments.exp_dax.run_experiment --mode vanilla

# Girsanov sweep with 2 controllers (automatic):
python -m v3.experiments.exp_dax.run_experiment --mode girsanov

# Everything (vanilla + girsanov + SLiCE):
python -m v3.experiments.exp_dax.run_experiment --mode both --include-slice
```

Common flags: `--mode {vanilla,girsanov,both}`, `--n-seeds N`,
`--include-neural-sde`, `--include-slice`,
`--start-girsanov-after-epoch K`, `--girsanov-batch-size-q B`.

## Outputs

For each block the run script saves:

* `all_losses_<block>.csv` — one row per (config, seed).
* `summary_<block>.csv` — one row per identity with `avg_<metric>` and
  `std_<metric>` across seeds; std is `NaN` when only one seed was run.
* `best_hyperparams_<block>.{json,pkl}` — the best config per identity.

## ESS bookkeeping

`final_ess_put` and `final_ess_call` are evaluated on the eval batch and
should be divided by `eval_batch_size` to get the per-controller ESS
ratio. The tqdm progress bar shows the *training-time* ESS, which is
divided by `girsanov_batch_size_q`. See `MODEL.md` §5 for the SNIS
definition.
