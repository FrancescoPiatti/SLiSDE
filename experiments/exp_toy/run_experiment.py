#!/usr/bin/env python
"""
exp_toy — long-path (default 2048 steps) toy-dataset orchestrator.

Two trainer modes:

* ``--mode vanilla`` (default): runs :class:`VanillaTrainer` on the
  ``exp1`` YAML section (4-component MSE loss, no Girsanov).
* ``--mode girsanov``: runs :class:`GirsanovTrainer` on the
  ``girsanov`` YAML section.  The trainer warm-starts under P for
  ``--start-girsanov-after-epoch`` epochs, then switches to the
  dual-batch shared-prefix forward pass with a self-normalised IS
  estimator on the barrier loss.
* ``--mode both``: runs both blocks back-to-back (separate result
  CSVs).

Usage:
    python -m v3.experiments.exp_toy.run_experiment --quick
    python -m v3.experiments.exp_toy.run_experiment --mode girsanov
    python -m v3.experiments.exp_toy.run_experiment \\
        --yaml v2/experiments/exp_toy/grid_search.yaml --mode both
"""
import argparse
import csv
import json
import math
import os
import pickle
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime

import numpy as np

from v3.datasets.prepare import load_toy
from v3.experiments.configs import (
    TrainConfig, build_slisde_grid, build_neural_sde_grid, build_slice_grid,
    model_identity_key,
)
from v3.experiments.exp_toy.trainer_vanilla import VanillaTrainer
from v3.experiments.exp_toy.trainer_girsanov import GirsanovTrainer
from v3.src.models.model import SLiSDEModel
from v3.benchmarks.slice_model import SLiCEStack

DEFAULT_MODEL_CLS = SLiSDEModel


METADATA_COLUMNS = (
    "num_params", "dim", "norm_type", "noise_rho", "time_dependent_vector_fields",
    "augment_brownian_with_time", "noise_dim", "lora_rank",
)


def _count_params(obj) -> int:
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
    *, tag, configs, trainer_cls, trainer_extra_kwargs,
    tc, dataset, results_dir, n_seeds, model_cls,
    rank_key="final_loss",
):
    """Train all configs, save all_losses CSV, summary CSV, and
    best_hyperparams JSON/PKL — mirrors the dax3 output format so
    downstream analysis tooling can ingest toy3 + dax3 uniformly.

    ``rank_key`` selects the metric used to pick the best config per
    identity (default ``final_loss`` — toy3 has no train/eval mask split,
    so ``final_eval_loss`` is unavailable here).
    """
    all_rows = []
    total = len(configs) * n_seeds
    run_idx = 0
    for cfg_name, cfg in configs.items():
        identity = model_identity_key(cfg, include_girsanov=False)
        for seed_idx in range(n_seeds):
            seed = 42 + seed_idx
            run_idx += 1
            print(f"\n[{tag}][{run_idx}/{total}] {cfg_name} seed={seed}")
            kwargs = dict(
                model_config=cfg, train_config=tc,
                dataset=dataset, config_name=cfg_name, model_cls=model_cls,
            )
            kwargs.update(trainer_extra_kwargs)
            trainer = trainer_cls(**kwargs)
            result = trainer.train(seed=seed)
            fm = result["final_metrics"]
            cfg_kwargs = asdict(cfg) if hasattr(cfg, "__dataclass_fields__") else vars(cfg)
            num_params = _count_params(result.get("params"))
            row = {
                "config_name": cfg_name, "seed": seed,
                "identity": identity, "config_kwargs": cfg_kwargs,
            }
            row.update(_metadata_row(cfg, num_params))
            for k, v in fm.items():
                row[k] = v
            all_rows.append(row)

    if not all_rows:
        print(f"[{tag}] no configs — skipping aggregation")
        return None

    metric_keys = sorted({k for r in all_rows for k in r if k.startswith("final_")})
    all_csv = os.path.join(results_dir, f"all_losses_{tag}.csv")
    fieldnames = (
        ["config_name", "seed", "identity"]
        + list(METADATA_COLUMNS) + metric_keys
    )
    with open(all_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"[{tag}] all losses saved to {all_csv}")

    # ---- Best per identity (mirrors dax3 _run_and_save) ----
    cfg_avg = defaultdict(list)
    cfg_info = {}
    for r in all_rows:
        cfg_avg[r["config_name"]].append(
            r.get(rank_key, r.get("final_loss", float("nan")))
        )
        cfg_info[r["config_name"]] = r

    def _safe_std(vals):
        """Sample std of a list; NaN when fewer than 2 finite values are present."""
        return float(np.std(vals, ddof=0)) if len(vals) >= 2 else float("nan")

    best = {}
    for cfg_name, losses in cfg_avg.items():
        finite = [x for x in losses if isinstance(x, (int, float)) and math.isfinite(x)]
        avg = float(np.mean(finite)) if finite else float("nan")
        info = cfg_info[cfg_name]
        identity = info["identity"]
        if (identity not in best
                or (math.isfinite(avg) and avg < best[identity]["avg_eval_loss"])):
            entry = {
                "config_name": cfg_name,
                "avg_eval_loss": avg,
                "std_eval_loss": _safe_std(finite),
                "n_seeds": len(losses),
                "config_kwargs": info["config_kwargs"],
            }
            # Per-metric mean and std across seeds.  std is NaN when only
            # one seed is available (the user-requested convention).
            for k in metric_keys:
                vals = [r[k] for r in all_rows
                        if r["config_name"] == cfg_name and k in r
                        and isinstance(r[k], (int, float))
                        and math.isfinite(r[k])]
                entry[f"avg_{k}"] = float(np.mean(vals)) if vals else float("nan")
                entry[f"std_{k}"] = _safe_std(vals)
            entry["avg_train_loss"] = entry.get(
                "avg_final_train_loss", entry.get("avg_final_loss", float("nan"))
            )
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
    print(f"[{tag}] best hyperparams saved to "
          f"best_hyperparams_{tag}.json/.pkl ({len(best_hp)} identities)")

    return best


def parse_args():
    p = argparse.ArgumentParser(
        description="exp_toy — long-path (2048-step) toy dataset, vanilla SLiSDE/NSDE."
    )
    p.add_argument("--quick", action="store_true")
    p.add_argument("--yaml", default=os.path.join(os.path.dirname(__file__), "grid_search.yaml"))
    p.add_argument("--include-neural-sde", action="store_true")
    p.add_argument(
        "--include-slice", action="store_true",
        help="Run the SLiCE benchmark (datasig-ac-uk/slices) as a separate "
             "block.  Uses VanillaTrainer (no Girsanov) with model_cls=SLiCEStack."
    )
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--num-epochs", type=int, default=2000)
    # toy3 default batch is much bigger than toy2 (256) — long paths benefit
    # from larger MC batches.
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--eval-batch-size", type=int, default=4096)
    p.add_argument("--barrier-batch-size", type=int, default=2048)
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument(
        "--mode", choices=["vanilla", "girsanov", "both"], default="vanilla",
        help="Which trainer block(s) to run.",
    )
    p.add_argument(
        "--start-girsanov-after-epoch", type=int, default=200,
        help="Number of P-only warm-up epochs before activating Girsanov.",
    )
    p.add_argument(
        "--girsanov-batch-size-q", type=int, default=256,
        help="Number of additional Q-paths per dual-batch step.",
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
    results_dir = args.output_dir or os.path.join("results", f"exp_toy_{ts}")
    os.makedirs(results_dir, exist_ok=True)

    print("Loading toy3 dataset (2048-step paths)...")
    dataset = load_toy()
    print(f"  y0={dataset.y0}, payoffs={dataset.target_payoffs.shape}, "
          f"barriers={dataset.target_barrier_probs.shape}, "
          f"maturity_indices={dataset.maturity_indices.tolist()}")

    tc = TrainConfig(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        barrier_batch_size=args.barrier_batch_size,
        n_seeds=args.n_seeds,
        use_wandb=not args.no_wandb,
        wandb_project="slisde-exp-toy3",
        start_girsanov_after_epoch=args.start_girsanov_after_epoch,
        girsanov_batch_size_q=args.girsanov_batch_size_q,
    )

    with open(os.path.join(results_dir, "run_config.json"), "w") as f:
        json.dump({
            "model_cls": DEFAULT_MODEL_CLS.__name__,
            "include_neural_sde": args.include_neural_sde,
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "barrier_batch_size": args.barrier_batch_size,
            "n_seeds": args.n_seeds,
        }, f, indent=2)

    if args.mode in ("vanilla", "both"):
        print("\n" + "=" * 60)
        print(f"BLOCK vanilla — vanilla SLiSDE/NSDE ({DEFAULT_MODEL_CLS.__name__})")
        print("=" * 60)
        cfgs = build_slisde_grid(args.yaml, section="vanilla")
        if args.include_neural_sde:
            cfgs.update(build_neural_sde_grid(args.yaml, section="neural_sde"))
        print(f"{len(cfgs)} configs")
        _run_and_save(
            tag="vanilla", configs=cfgs,
            trainer_cls=VanillaTrainer, trainer_extra_kwargs={},
            tc=tc, dataset=dataset, results_dir=results_dir,
            n_seeds=args.n_seeds, model_cls=DEFAULT_MODEL_CLS,
        )

    if args.include_slice:
        print("\n" + "=" * 60)
        print("BLOCK slice — SLiCE benchmark (datasig-ac-uk/slices)")
        print("=" * 60)
        slice_cfgs = build_slice_grid(args.yaml, section="slice")
        print(f"{len(slice_cfgs)} configs")
        if slice_cfgs:
            _run_and_save(
                tag="slice", configs=slice_cfgs,
                trainer_cls=VanillaTrainer, trainer_extra_kwargs={},
                tc=tc, dataset=dataset, results_dir=results_dir,
                n_seeds=args.n_seeds, model_cls=SLiCEStack,
            )
        else:
            print("[slice] no SLiCE configs in YAML — skipping")

    if args.mode in ("girsanov", "both"):
        print("\n" + "=" * 60)
        print("BLOCK girsanov — last-layer Girsanov SLiSDE")
        print("=" * 60)
        gir_cfgs = build_slisde_grid(args.yaml, section="girsanov")
        # Drop any non-Girsanov entries that snuck through (defensive).
        gir_cfgs = {k: v for k, v in gir_cfgs.items()
                    if getattr(v, "use_girsanov", False)}
        print(f"{len(gir_cfgs)} configs")
        if gir_cfgs:
            _run_and_save(
                tag="girsanov", configs=gir_cfgs,
                trainer_cls=GirsanovTrainer, trainer_extra_kwargs={},
                tc=tc, dataset=dataset, results_dir=results_dir,
                n_seeds=args.n_seeds, model_cls=DEFAULT_MODEL_CLS,
            )
        else:
            print("[girsanov] no Girsanov configs in YAML — skipping")

    print(f"\nexp_toy COMPLETE. Results in: {results_dir}")


if __name__ == "__main__":
    main()
