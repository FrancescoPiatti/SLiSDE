#!/usr/bin/env python
"""Prepare and pickle the v3 datasets (``toy``, ``dax``).

Both datasets target ``model_n_steps = 2048`` paths on $[0, T]$ with $T = 1$.
``toy`` is a synthetic 3-component non-linear SDE used for path-functional
calibration (payoffs, running max, squared average, barrier probabilities).
``dax`` is a dense DAX option-surface dataset (25 calls + 25 puts +
20 OTM puts + 15 OTM calls + 12 digital puts) loaded from a quote CSV and
normalised by the spot price.

Usage::

    python -m v3.datasets.prepare              # prepare both toy + dax
    python -m v3.datasets.prepare --toy        # toy only
    python -m v3.datasets.prepare --dax        # dax only
    python -m v3.datasets.prepare --n-mc 500000  # more MC paths (toy)

Pickled files land in::

    v3/datasets/toy/toy_dataset.pkl
    v3/datasets/dax/dax_dataset.pkl

The matching ``load_toy()`` / ``load_dax()`` helpers reload them.
"""
import argparse
import os
import pickle

import jax.random as jr


# v3 baseline: 2048 steps for both datasets.
TOY_N_STEPS = 2048
DAX_N_STEPS = 2048


def prepare_toy(n_mc_paths: int = 200_000, seed: int = 0,
                model_n_steps: int = TOY_N_STEPS):
    """Generate and pickle the toy calibration dataset (long paths)."""
    from v3.datasets.toy.toy_dataset import generate_dataset

    print("=" * 60)
    print(f"Preparing toy calibration dataset  (n_steps={model_n_steps})")
    print(f"  n_mc_paths = {n_mc_paths:,}")
    print(f"  seed       = {seed}")
    print("=" * 60)

    dataset = generate_dataset(
        n_mc_paths=n_mc_paths,
        model_n_steps=model_n_steps,
        key=jr.PRNGKey(seed),
    )

    out_path = os.path.join(os.path.dirname(__file__), "toy", "toy_dataset.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(dataset, f)
    print(f"\nSaved to {out_path}  ({os.path.getsize(out_path) / 1024:.1f} KB)")
    return dataset


def prepare_dax(csv_path: str = None, spot: float = 25280.0,
                model_n_steps: int = DAX_N_STEPS):
    """Load and pickle the dax options dataset on the model time grid."""
    from v3.datasets.dax.dax2_dataset import load_dax2_options

    print("=" * 60)
    print(f"Preparing dax (dense option surface) dataset")
    print(f"  spot         = {spot}")
    print(f"  model_n_steps= {model_n_steps}")
    print("=" * 60)

    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(__file__), "dax", "dax_options.csv")

    dataset = load_dax2_options(
        csv_path=csv_path, spot=spot, model_n_steps=model_n_steps,
    )

    out_path = os.path.join(os.path.dirname(__file__), "dax", "dax_dataset.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(dataset, f)
    print(f"\nSaved to {out_path}  ({os.path.getsize(out_path) / 1024:.1f} KB)")
    return dataset


def load_toy():
    """Load the pickled toy dataset (raises if not prepared)."""
    pkl_path = os.path.join(os.path.dirname(__file__), "toy", "toy_dataset.pkl")
    if not os.path.isfile(pkl_path):
        raise FileNotFoundError(
            f"toy dataset not found at {pkl_path}. "
            "Run: python -m v3.datasets.prepare --toy"
        )
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def load_dax():
    """Load the pickled dax dataset (raises if not prepared)."""
    pkl_path = os.path.join(os.path.dirname(__file__), "dax", "dax_dataset.pkl")
    if not os.path.isfile(pkl_path):
        raise FileNotFoundError(
            f"dax dataset not found at {pkl_path}. "
            "Run: python -m v3.datasets.prepare --dax"
        )
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(description="Prepare v3 (toy, dax) datasets")
    parser.add_argument("--toy", action="store_true", help="Prepare toy only")
    parser.add_argument("--dax", action="store_true", help="Prepare dax only")
    parser.add_argument("--n-mc", type=int, default=200_000,
                        help="Number of MC paths for toy dataset")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for toy")
    parser.add_argument("--spot", type=float, default=25280.0, help="DAX spot price")
    parser.add_argument("--csv-path", type=str, default=None,
                        help="Path to dax_options.csv")
    parser.add_argument(
        "--n-steps", type=int, default=None,
        help=f"Override model_n_steps "
             f"(default toy={TOY_N_STEPS}, dax={DAX_N_STEPS})",
    )
    args = parser.parse_args()

    do_all = not args.toy and not args.dax
    if args.toy or do_all:
        prepare_toy(
            n_mc_paths=args.n_mc, seed=args.seed,
            model_n_steps=args.n_steps or TOY_N_STEPS,
        )
        print()
    if args.dax or do_all:
        prepare_dax(
            csv_path=args.csv_path, spot=args.spot,
            model_n_steps=args.n_steps or DAX_N_STEPS,
        )
        print()
    print("Done. Use v3.datasets.prepare.load_toy() / load_dax().")


if __name__ == "__main__":
    main()
