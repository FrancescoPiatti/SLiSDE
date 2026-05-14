# experiments/exp1/trainer.py
"""
Trainer for exp1: no-Girsanov baseline with barrier loss (Flax NNX).

Loss = weighted avg of (L_payoff + L_runmax + L_sqavg + L_barrier)
"""
import time
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from flax import nnx
from tqdm import tqdm

from v3.src.config import SLiSDEConfig
from v3.src.models.model import SLiSDEModel
from v3.benchmarks.neural_sde import NeuralSDEStack
from v3.benchmarks.neural_sde_config import NeuralSDEConfig
from v3.benchmarks.slice_model import SLiCEStack
from v3.benchmarks.slice_config import SLiCEConfig
from v3.experiments.configs import TrainConfig
from v3.datasets import CalibrationDataset


def _build_model(cfg, rngs, model_cls=None):
    """Build the model that matches ``cfg``.

    Dispatches on config type so the same trainer can run SLiSDE,
    Neural-SDE and SLiCE baselines without touching the call sites.
    Passing ``model_cls`` explicitly overrides this dispatch (legacy).
    """
    if model_cls is not None:
        return model_cls(config=cfg, rngs=rngs)
    if isinstance(cfg, NeuralSDEConfig):
        return NeuralSDEStack(config=cfg, rngs=rngs)
    if isinstance(cfg, SLiCEConfig):
        return SLiCEStack(config=cfg, rngs=rngs)
    return SLiSDEModel(config=cfg, rngs=rngs)


class VanillaTrainer:
    """Trains a model with standard losses + naive barrier loss (no Girsanov)."""

    def __init__(
        self,
        model_config,
        train_config: TrainConfig,
        dataset: CalibrationDataset,
        config_name: str = "default",
        model_cls=None,
    ) -> None:
        self.model_config = model_config
        self.train_config = train_config
        self.dataset = dataset
        self.config_name = config_name
        self.model_cls = model_cls  # SLiSDEModel / SLiSDEModel2; None -> default
        self.is_neural_sde = isinstance(model_config, NeuralSDEConfig)

        self.target_payoffs = dataset.target_payoffs
        self.target_running_max = dataset.target_running_max
        self.target_squared_avg = dataset.target_squared_avg
        self.target_barrier_probs = dataset.target_barrier_probs
        self.barriers = dataset.barriers
        self.strikes = dataset.strikes
        self.mat_indices = dataset.maturity_indices
        self.n_steps = model_config.n_steps
        self.y0 = dataset.y0

    def _compute_loss(self, model, batch_size):
        """Compute 4-component loss. Must be called inside nnx.value_and_grad."""
        target_payoffs = self.target_payoffs
        target_running_max = self.target_running_max
        target_squared_avg = self.target_squared_avg
        target_barrier_probs = self.target_barrier_probs
        barriers = self.barriers
        strikes = self.strikes
        mat_indices = self.mat_indices
        n_steps = self.n_steps
        clip_z = self.train_config.clip_z
        y0 = self.y0
        barrier_weight = self.train_config.barrier_weight

        paths = model(batch_size=batch_size, return_path=True, y0=y0)
        z = jnp.clip(paths[:, :, 0], -clip_z, clip_z)

        z_at_mats = z[:, mat_indices]
        payoffs = jax.nn.relu(z_at_mats[:, None, :] - strikes[None, :, None])
        pred_payoffs = jnp.mean(payoffs, axis=0)

        cummax = jax.lax.associative_scan(jnp.maximum, z, axis=1)
        pred_runmax = jnp.mean(cummax[:, mat_indices], axis=0)

        cumsq = jnp.cumsum(z ** 2, axis=1)
        divisors = jnp.arange(1, n_steps + 2, dtype=z.dtype)[None, :]
        pred_sqavg = jnp.mean((cumsq / divisors)[:, mat_indices], axis=0)

        runmax_at_mats = cummax[:, mat_indices]
        hit = (runmax_at_mats[:, None, :] >= barriers[None, :, None]).astype(z.dtype)
        pred_barrier = jnp.mean(hit, axis=0)

        loss_payoffs = jnp.mean((pred_payoffs - target_payoffs) ** 2)
        loss_runmax = jnp.mean((pred_runmax - target_running_max) ** 2)
        loss_sqavg = jnp.mean((pred_sqavg - target_squared_avg) ** 2)
        loss_barrier = jnp.mean((pred_barrier - target_barrier_probs) ** 2)

        loss_std = (loss_payoffs + loss_runmax + loss_sqavg) / 3.0
        loss = (loss_std + barrier_weight * loss_barrier) / (1.0 + barrier_weight)

        metrics = {
            "loss": loss, "loss_payoffs": loss_payoffs,
            "loss_runmax": loss_runmax, "loss_sqavg": loss_sqavg,
            "loss_barrier": loss_barrier,
            "pred_payoffs": pred_payoffs, "pred_runmax": pred_runmax,
            "pred_sqavg": pred_sqavg, "pred_barrier": pred_barrier,
        }
        return loss, metrics

    def train(self, seed: int) -> dict:
        tc = self.train_config

        run = None
        if tc.use_wandb:
            import wandb
            run = wandb.init(
                project=tc.wandb_project,
                config={"seed": seed, "config_name": self.config_name},
                name=f"vanilla_{self.config_name}_seed{seed}",
                group=f"vanilla_{self.config_name}", reinit=True,
            )

        rngs = nnx.Rngs(params=seed, init=seed + 1000, noise=seed + 2000)
        model = _build_model(self.model_config, rngs, model_cls=self.model_cls)

        warmup_steps = min(tc.warmup_steps, tc.num_epochs // 2)
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=tc.end_lr, peak_value=tc.peak_lr,
            warmup_steps=warmup_steps, decay_steps=tc.num_epochs, end_value=tc.end_lr,
        )
        opt = optax.chain(
            optax.clip_by_global_norm(tc.grad_clip_norm),
            optax.adamw(learning_rate=schedule, weight_decay=tc.weight_decay),
        )
        optimizer = nnx.Optimizer(model, opt, wrt=nnx.Param)

        # Capture self for closure
        trainer = self
        train_bs = tc.batch_size
        eval_bs = tc.eval_batch_size

        @nnx.jit
        def train_step(model, optimizer):
            def loss_fn(model):
                return trainer._compute_loss(model, train_bs)
            (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
            optimizer.update(model, grads)
            grad_norm = optax.global_norm(grads)
            return loss, metrics, grad_norm

        @nnx.jit
        def eval_step(model):
            return trainer._compute_loss(model, eval_bs)

        history = {
            "train_loss": [], "eval_loss": [], "lr": [], "grad_norm": [],
            "loss_payoffs": [], "loss_runmax": [], "loss_sqavg": [],
            "loss_barrier": [],
        }

        pbar = tqdm(range(tc.num_epochs), desc=f"exp1 {self.config_name} seed={seed}")
        t_start = time.time()

        for step in pbar:
            loss, metrics, grad_norm = train_step(model, optimizer)
            lr = schedule(step)
            loss_val = float(loss)
            history["train_loss"].append(loss_val)
            history["lr"].append(float(lr))
            history["grad_norm"].append(float(grad_norm))

            if step % tc.log_every == 0 or step == tc.num_epochs - 1:
                pbar.set_postfix(
                    loss=f"{loss_val:.6f}",
                    bar=f"{float(metrics['loss_barrier']):.6f}",
                    lr=f"{float(lr):.2e}",
                )
                if tc.use_wandb and run is not None:
                    import wandb
                    wandb.log({"train/loss": loss_val, "step": step})

            if step % tc.eval_every == 0 or step == tc.num_epochs - 1:
                eval_loss, eval_metrics = eval_step(model)
                history["eval_loss"].append(float(eval_loss))
                history["loss_payoffs"].append(float(eval_metrics["loss_payoffs"]))
                history["loss_runmax"].append(float(eval_metrics["loss_runmax"]))
                history["loss_sqavg"].append(float(eval_metrics["loss_sqavg"]))
                history["loss_barrier"].append(float(eval_metrics["loss_barrier"]))

        elapsed = time.time() - t_start
        print(f"\nexp1 {self.config_name} seed={seed} done in {elapsed:.1f}s  "
              f"(final loss={history['train_loss'][-1]:.6f})")

        eval_loss, eval_metrics = eval_step(model)
        _, params, _ = nnx.split(model, nnx.Param, nnx.RngState)

        final_result = {
            "final_loss": float(eval_loss),
            "final_mae_payoffs": float(jnp.mean(jnp.abs(eval_metrics["pred_payoffs"] - self.target_payoffs))),
            "final_mae_runmax": float(jnp.mean(jnp.abs(eval_metrics["pred_runmax"] - self.target_running_max))),
            "final_mae_sqavg": float(jnp.mean(jnp.abs(eval_metrics["pred_sqavg"] - self.target_squared_avg))),
            "final_mae_barrier": float(jnp.mean(jnp.abs(eval_metrics["pred_barrier"] - self.target_barrier_probs))),
        }

        if tc.use_wandb and run is not None:
            import wandb
            wandb.summary.update({"final/loss": final_result["final_loss"], "elapsed_seconds": elapsed})
            wandb.finish()

        return {"params": params, "history": history, "final_metrics": final_result,
                "config_name": self.config_name, "seed": seed}

    def generate_paths(self, params, n_paths: int = 50, seed: int = 999):
        rngs = nnx.Rngs(params=0, init=seed, noise=seed + 1)
        model = _build_model(self.model_config, rngs, model_cls=self.model_cls)
        graphdef, _, rng_state = nnx.split(model, nnx.Param, nnx.RngState)
        model = nnx.merge(graphdef, params, rng_state)
        return model(batch_size=n_paths, return_path=True, y0=self.y0)[:, :, 0]
