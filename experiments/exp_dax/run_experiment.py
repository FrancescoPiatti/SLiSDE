#!/usr/bin/env python
"""
exp_dax — long-path (default 2048-step) DAX options calibration, masked strikes.

Same train/eval strike-split design as exp_dax2 (dense DAX grid, masked
strike loss, overfit-gap diagnostic) but with:
  * 2048-step paths (vs. 256 in exp_dax2),
  * larger batch sizes (long paths benefit from bigger MC batches),
  * v2's vanilla ``SLiSDEModel`` only (no Girsanov, no model2).

Reuses:
  - ``v3.experiments.exp_dax.trainer_vanilla.VanillaTrainer``
  - ``v3.experiments.exp_dax.masking.create_strike_masks``

Usage:
    python -m v3.experiments.exp_dax.run_experiment --quick
    python -m v3.experiments.exp_dax.run_experiment \\
        --yaml v2/experiments/exp_dax/grid_search.yaml
"""
import argparse
import csv
import json
import os
import pickle
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime

import numpy as np

from v3.datasets.prepare import load_dax
from v3.experiments.configs import (
    TrainConfig, build_slisde_grid, build_neural_sde_grid, build_slice_grid,
    model_identity_key,
)
from v3.experiments.exp_dax.masking import create_strike_masks
from v3.experiments.exp_dax.trainer_vanilla import VanillaTrainer
from v3.experiments.exp_dax.trainer_girsanov import GirsanovTrainer
from v3.src.models.model import SLiSDEModel
from v3.benchmarks.slice_model import SLiCEStack

DEFAULT_MODEL_CLS = SLiSDEModel


# ─── Shared metadata columns added to every all_losses_*.csv row ──
METADATA_COLUMNS = (
    "num_params",
    "dim",
    "norm_type",
    "noise_rho",
    "time_dependent_vector_fields",
    "augment_brownian_with_time",
    "noise_dim",
    "lora_rank",
)


def _count_params(obj) -> int:
    """Return total parameter count from an nnx.Module or an nnx.State tree."""
    try:
        import jax
        from flax import nnx

        if obj is None:
            return 0
        if isinstance(obj, nnx.Module):
            _, state = nnx.split(obj, nnx.Param, ...)
            tree = state
        else:
            tree = obj
        leaves = jax.tree_util.tree_leaves(tree)
        total = 0
        for leaf in leaves:
            if hasattr(leaf, "size"):
                total += int(leaf.size)
            elif hasattr(leaf, "shape"):
                sz = 1
                for d in leaf.shape:
                    sz *= int(d)
                total += sz
        return int(total)
    except Exception:
        return 0


def _metadata_row(cfg, num_params: int) -> dict:
    def _get(name, default):
        return getattr(cfg, name, default)
    return {
        "num_params": num_params,
        "dim": _get("dim", 0),
        "norm_type": _get("norm_type", "none"),
        "noise_rho": _get("noise_rho", 0.0),
        "time_dependent_vector_fields": _get("time_dependent_vector_fields", "none"),
        "augment_brownian_with_time": _get("augment_brownian_with_time", False),
        "noise_dim": _get("noise_dim", 0),
        "lora_rank": _get("lora_rank", 0),
    }


def _run_and_save(
    *,
    configs, tc, dataset, masks, results_dir, tag,
    trainer_cls, n_seeds, model_cls,
    save_training_losses: bool = False,
):
    """Train all configs, save results with train/eval split metrics.

    When ``save_training_losses=True`` the per-step training-loss curve
    (and any other history channels the trainer reports) for every
    (config_name, seed) run is written to
    ``results_dir/training_losses_<tag>.json`` at the end of the block.
    """
    all_rows = []
    histories = {} if save_training_losses else None
    total = len(configs) * n_seeds
    run_idx = 0

    for cfg_name, cfg in configs.items():
        identity = model_identity_key(cfg, include_girsanov=False)
        for seed_idx in range(n_seeds):
            seed = 42 + seed_idx
            run_idx += 1
            print(f"\n[{tag}][{run_idx}/{total}] {cfg_name} seed={seed}")

            trainer = trainer_cls(
                model_config=cfg, train_config=tc,
                dataset=dataset, masks=masks, config_name=cfg_name,
                model_cls=model_cls,
            )
            result = trainer.train(seed=seed)

            fm = result["final_metrics"]
            cfg_kwargs = asdict(cfg) if hasattr(cfg, "__dataclass_fields__") else {}
            params_obj = result.get("params") if isinstance(result, dict) else None
            num_params = _count_params(params_obj)
            row = {
                "config_name": cfg_name,
                "seed": seed,
                "identity": identity,
                "config_kwargs": cfg_kwargs,
            }
            row.update(_metadata_row(cfg, num_params))
            for k, v in fm.items():
                row[k] = v
            all_rows.append(row)

            if histories is not None:
                hist = result.get("history", {}) if isinstance(result, dict) else {}
                # Coerce to plain Python lists of floats so the JSON dump
                # is robust to JAX/numpy scalar types in the history.
                clean = {
                    k: [float(x) for x in v]
                    for k, v in hist.items()
                    if isinstance(v, (list, tuple))
                }
                histories.setdefault(cfg_name, {})[str(seed)] = clean

    if not all_rows:
        print(f"[{tag}] no configs — skipping aggregation")
        return None

    metric_keys = sorted({k for r in all_rows for k in r if k.startswith("final_")})
    all_csv = os.path.join(results_dir, f"all_losses_{tag}.csv")
    fieldnames = (
        ["config_name", "seed", "identity"]
        + list(METADATA_COLUMNS)
        + metric_keys
    )
    with open(all_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"[{tag}] all losses saved to {all_csv}")

    rank_key = "final_eval_loss"
    cfg_avg = defaultdict(list)
    cfg_info = {}
    for r in all_rows:
        cfg_avg[r["config_name"]].append(r.get(rank_key, r.get("final_train_loss", 0.0)))
        cfg_info[r["config_name"]] = r

    def _safe_std(vals):
        """Sample std of a list; NaN when fewer than 2 finite values are present."""
        return float(np.std(vals, ddof=0)) if len(vals) >= 2 else float("nan")

    best = {}
    for cfg_name, losses in cfg_avg.items():
        finite_losses = [x for x in losses if isinstance(x, (int, float))]
        avg = float(np.mean(finite_losses)) if finite_losses else float("nan")
        info = cfg_info[cfg_name]
        identity = info["identity"]
        if identity not in best or avg < best[identity]["avg_eval_loss"]:
            entry = {
                "config_name": cfg_name,
                "avg_eval_loss": avg,
                "std_eval_loss": _safe_std(finite_losses),
                "n_seeds": len(losses),
                "config_kwargs": info["config_kwargs"],
            }
            # Per-metric mean and std across seeds.  std is NaN if only one
            # seed (the user-requested convention).
            for k in metric_keys:
                vals = [r[k] for r in all_rows if r["config_name"] == cfg_name and k in r
                        and isinstance(r[k], (int, float))]
                entry[f"avg_{k}"] = float(np.mean(vals)) if vals else float("nan")
                entry[f"std_{k}"] = _safe_std(vals)
            entry["avg_train_loss"] = entry.get("avg_final_train_loss", float("nan"))
            entry["avg_overfit_gap"] = entry.get("avg_final_overfit_gap", float("nan"))
            best[identity] = entry

    summary_csv = os.path.join(results_dir, f"summary_{tag}.csv")
    summary_fieldnames = (
        ["identity", "config_name", "avg_train_loss", "avg_eval_loss",
         "avg_overfit_gap", "std_eval_loss", "n_seeds"]
        + [f"avg_{k}" for k in metric_keys]
        + [f"std_{k}" for k in metric_keys]
    )
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fieldnames)
        w.writeheader()
        for identity in sorted(best.keys()):
            e = best[identity]
            row = {
                "identity": identity,
                "config_name": e["config_name"],
                "avg_train_loss": f"{e['avg_train_loss']:.8f}",
                "avg_eval_loss":  f"{e['avg_eval_loss']:.8f}",
                "avg_overfit_gap": f"{e['avg_overfit_gap']:.8f}",
                "std_eval_loss":  f"{e['std_eval_loss']:.8f}",
                "n_seeds": e["n_seeds"],
            }
            for k in metric_keys:
                row[f"avg_{k}"] = f"{e.get(f'avg_{k}', float('nan')):.8f}"
                row[f"std_{k}"] = f"{e.get(f'std_{k}', float('nan')):.8f}"
            w.writerow(row)
    print(f"[{tag}] summary saved to {summary_csv}")

    best_hp = {
        identity: {
            "config_name": e["config_name"],
            "avg_eval_loss": e["avg_eval_loss"],
            "avg_overfit_gap": e["avg_overfit_gap"],
            "config_kwargs": e["config_kwargs"],
        }
        for identity, e in best.items()
    }
    with open(os.path.join(results_dir, f"best_hyperparams_{tag}.json"), "w") as f:
        json.dump(best_hp, f, indent=2, default=str)
    with open(os.path.join(results_dir, f"best_hyperparams_{tag}.pkl"), "wb") as f:
        pickle.dump(best_hp, f)

    if histories is not None:
        losses_path = os.path.join(results_dir, f"training_losses_{tag}.json")
        with open(losses_path, "w") as f:
            json.dump(histories, f)
        n_runs = sum(len(v) for v in histories.values())
        print(f"[{tag}] training-loss curves saved to {losses_path} "
              f"({n_runs} runs)")

    return best


def parse_args():
    p = argparse.ArgumentParser(
        description="exp_dax — long-path (2048-step) DAX calibration, vanilla SLiSDE/NSDE."
    )
    p.add_argument("--quick", action="store_true")
    p.add_argument("--yaml", default=os.path.join(os.path.dirname(__file__), "grid_search.yaml"))
    p.add_argument("--include-neural-sde", action="store_true")
    p.add_argument(
        "--include-slice", action="store_true",
        help="Run the SLiCE benchmark (datasig-ac-uk/slices) as a separate "
             "block.  Uses VanillaTrainer with model_cls=SLiCEStack."
    )
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--num-epochs", type=int, default=2000)
    # exp_dax: long paths benefit from larger MC batches relative to exp_dax2.
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--eval-batch-size", type=int, default=4096)
    p.add_argument("--barrier-batch-size", type=int, default=2048)
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--eval-frac", type=float, default=0.25)
    p.add_argument("--mask-seed", type=int, default=0)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument(
        "--mode", choices=["vanilla", "girsanov", "both"], default="vanilla",
        help="Which trainer block(s) to run.",
    )
    p.add_argument("--start-girsanov-after-epoch", type=int, default=200)
    p.add_argument("--girsanov-batch-size-q", type=int, default=256)
    p.add_argument(
        "--save-training-losses", action="store_true",
        help="Also write per-run training-loss curves (and other history "
             "channels) to training_losses_<tag>.json for each block.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.num_epochs = 50
        args.n_seeds = 1
        args.no_wandb = True
        args.batch_size = 64
        args.eval_batch_size = 256
        args.barrier_batch_size = 128
        args.girsanov_batch_size_q = 32
        args.start_girsanov_after_epoch = 10
        print("=== QUICK MODE ===")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = args.output_dir or os.path.join("results", f"exp_dax_{ts}")
    os.makedirs(results_dir, exist_ok=True)

    print("Loading dax3 dataset (2048-step paths)...")
    dataset = load_dax()

    print(f"\nCreating train/eval split (eval_frac={args.eval_frac}, seed={args.mask_seed})...")
    masks = create_strike_masks(dataset, eval_frac=args.eval_frac, seed=args.mask_seed)
    print(masks.summary())

    with open(os.path.join(results_dir, "mask_config.json"), "w") as f:
        json.dump({"eval_frac": args.eval_frac, "mask_seed": args.mask_seed}, f, indent=2)
    with open(os.path.join(results_dir, "run_config.json"), "w") as f:
        json.dump({
            "model_cls": DEFAULT_MODEL_CLS.__name__,
            "include_neural_sde": args.include_neural_sde,
            "eval_frac": args.eval_frac,
            "mask_seed": args.mask_seed,
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "barrier_batch_size": args.barrier_batch_size,
            "n_seeds": args.n_seeds,
        }, f, indent=2)

    tc = TrainConfig(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        barrier_batch_size=args.barrier_batch_size,
        n_seeds=args.n_seeds,
        use_wandb=not args.no_wandb,
        wandb_project="slisde-exp-dax3",
        start_girsanov_after_epoch=args.start_girsanov_after_epoch,
        girsanov_batch_size_q=args.girsanov_batch_size_q,
    )

    if args.mode in ("vanilla", "both"):
        print("\n" + "=" * 60)
        print(f"BLOCK vanilla — SLiSDE/NSDE ({DEFAULT_MODEL_CLS.__name__})")
        print("=" * 60)
        cfgs = build_slisde_grid(args.yaml, section="vanilla")
        if args.include_neural_sde:
            cfgs.update(build_neural_sde_grid(args.yaml, section="neural_sde"))
        print(f"{len(cfgs)} configs")
        _run_and_save(
            configs=cfgs, tc=tc, dataset=dataset, masks=masks,
            results_dir=results_dir, tag="vanilla",
            trainer_cls=VanillaTrainer, n_seeds=args.n_seeds,
            model_cls=DEFAULT_MODEL_CLS,
            save_training_losses=args.save_training_losses,
        )

    if args.include_slice:
        print("\n" + "=" * 60)
        print("BLOCK slice — SLiCE benchmark (datasig-ac-uk/slices)")
        print("=" * 60)
        slice_cfgs = build_slice_grid(args.yaml, section="slice")
        print(f"{len(slice_cfgs)} configs")
        if slice_cfgs:
            _run_and_save(
                configs=slice_cfgs, tc=tc, dataset=dataset, masks=masks,
                results_dir=results_dir, tag="slice",
                trainer_cls=VanillaTrainer, n_seeds=args.n_seeds,
                model_cls=SLiCEStack,
                save_training_losses=args.save_training_losses,
            )
        else:
            print("[slice] no SLiCE configs in YAML — skipping")

    if args.mode in ("girsanov", "both"):
        print("\n" + "=" * 60)
        print("BLOCK girsanov — last-layer-only Girsanov SLiSDE")
        print("=" * 60)
        gir_cfgs = build_slisde_grid(args.yaml, section="girsanov")
        gir_cfgs = {k: v for k, v in gir_cfgs.items()
                    if getattr(v, "use_girsanov", False)}
        # v3 dax girsanov always uses TWO controllers (OTM-put + OTM-call).
        # Pin num_girsanov=2 so the user does not have to remember to set it
        # in the YAML.
        for cfg in gir_cfgs.values():
            cfg.num_girsanov = 2
        print(f"{len(gir_cfgs)} configs")
        if gir_cfgs:
            _run_and_save(
                configs=gir_cfgs, tc=tc, dataset=dataset, masks=masks,
                results_dir=results_dir, tag="girsanov",
                trainer_cls=GirsanovTrainer, n_seeds=args.n_seeds,
                model_cls=DEFAULT_MODEL_CLS,
                save_training_losses=args.save_training_losses,
            )
        else:
            print("[girsanov] no Girsanov configs in YAML — skipping")

    print(f"\nexp_dax COMPLETE. Results in: {results_dir}")


if __name__ == "__main__":
    main()
