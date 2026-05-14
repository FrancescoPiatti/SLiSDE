# experiments/exp_dax2/trainer_vanilla.py
"""
DAX2 vanilla trainer with train/eval strike masking for overfitting detection.

Trains on a subset of strikes; evaluates on both train and held-out strikes.
"""
import time

import jax
import jax.numpy as jnp
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
from v3.datasets import DaxOptionsDataset
from v3.experiments.exp_dax.masking import StrikeMasks


def _build_model(cfg, rngs, model_cls=None):
    """Dispatch on config type to build SLiSDE / Neural-SDE / SLiCE."""
    if model_cls is not None:
        return model_cls(config=cfg, rngs=rngs)
    if isinstance(cfg, NeuralSDEConfig):
        return NeuralSDEStack(config=cfg, rngs=rngs)
    if isinstance(cfg, SLiCEConfig):
        return SLiCEStack(config=cfg, rngs=rngs)
    return SLiSDEModel(config=cfg, rngs=rngs)


def _masked_mse(pred, target, mask):
    """MSE over masked strikes. mask is (n_strikes,), broadcast over maturities."""
    diff2 = (pred - target) ** 2
    # mask shape: (n_strikes,) -> (n_strikes, 1) for broadcasting over maturities
    m = mask[:, None].astype(diff2.dtype)
    return jnp.sum(diff2 * m) / (jnp.sum(m) * target.shape[1] + 1e-12)


class VanillaTrainer:
    """Vanilla DAX2 trainer with train/eval strike split."""

    def __init__(
        self,
        model_config,
        train_config: TrainConfig,
        dataset: DaxOptionsDataset,
        masks: StrikeMasks,
        config_name: str = "default",
        w_call: float = 1.0,
        w_put: float = 1.0,
        w_otm: float = 1.0,
        w_otm_call: float = 1.0,
        w_digital: float = 2.0,
        model_cls=None,
    ):
        self.model_config = model_config
        self.train_config = train_config
        self.dataset = dataset
        self.masks = masks
        self.config_name = config_name
        self.model_cls = model_cls
        self.w_call = w_call
        self.w_put = w_put
        self.w_otm = w_otm
        self.w_otm_call = w_otm_call
        self.w_digital = w_digital

        self.call_strikes = dataset.call_strikes
        self.call_prices = dataset.call_prices
        self.put_strikes = dataset.put_strikes
        self.put_prices = dataset.put_prices
        self.otm_put_strikes = dataset.otm_put_strikes
        self.otm_put_prices = dataset.otm_put_prices
        self.otm_call_strikes = dataset.otm_call_strikes
        self.otm_call_prices = dataset.otm_call_prices
        self.digital_put_strikes = dataset.digital_put_strikes
        self.digital_put_prices = dataset.digital_put_prices
        self.mat_indices = dataset.maturity_indices
        self.y0 = dataset.y0

    def _compute_predictions(self, model, batch_size):
        """Forward pass: compute all predictions (shared by train and eval)."""
        clip_z = self.train_config.clip_z
        paths = model(batch_size=batch_size, return_path=True, y0=self.y0)
        Y = jnp.clip(paths[:, :, 0], -clip_z, clip_z)
        Y_at_mats = Y[:, self.mat_indices]

        pred_call = jnp.mean(jax.nn.relu(Y_at_mats[:, None, :] - self.call_strikes[None, :, None]), axis=0)
        pred_put = jnp.mean(jax.nn.relu(self.put_strikes[None, :, None] - Y_at_mats[:, None, :]), axis=0)
        pred_otm = jnp.mean(jax.nn.relu(self.otm_put_strikes[None, :, None] - Y_at_mats[:, None, :]), axis=0)
        pred_otm_call = jnp.mean(jax.nn.relu(Y_at_mats[:, None, :] - self.otm_call_strikes[None, :, None]), axis=0)

        sharpness = 50.0
        digital_indicator = jax.nn.sigmoid(sharpness * (self.digital_put_strikes[None, :, None] - Y_at_mats[:, None, :]))
        pred_digital = jnp.mean(digital_indicator, axis=0)

        return pred_call, pred_put, pred_otm, pred_otm_call, pred_digital

    def _compute_train_loss(self, model, batch_size):
        """Loss on train strikes only."""
        pred_call, pred_put, pred_otm, pred_otm_call, pred_digital = \
            self._compute_predictions(model, batch_size)
        masks = self.masks
        w_call, w_put, w_otm = self.w_call, self.w_put, self.w_otm
        w_otm_call, w_digital = self.w_otm_call, self.w_digital

        loss_call = _masked_mse(pred_call, self.call_prices, masks.call_train)
        loss_put = _masked_mse(pred_put, self.put_prices, masks.put_train)
        loss_otm = _masked_mse(pred_otm, self.otm_put_prices, masks.otm_put_train)
        loss_otm_call = _masked_mse(pred_otm_call, self.otm_call_prices, masks.otm_call_train)
        loss_digital = _masked_mse(pred_digital, self.digital_put_prices, masks.digital_train)

        total_w = w_call + w_put + w_otm + w_otm_call + w_digital
        loss = (w_call * loss_call + w_put * loss_put + w_otm * loss_otm
                + w_otm_call * loss_otm_call + w_digital * loss_digital) / total_w

        metrics = {
            "loss": loss, "loss_call": loss_call, "loss_put": loss_put,
            "loss_otm": loss_otm, "loss_otm_call": loss_otm_call,
            "loss_digital": loss_digital,
        }
        return loss, metrics

    def _compute_eval_metrics(self, model, batch_size):
        """Compute losses on both train and eval strikes."""
        pred_call, pred_put, pred_otm, pred_otm_call, pred_digital = \
            self._compute_predictions(model, batch_size)
        masks = self.masks

        def _both(pred, target, train_mask):
            train_mse = _masked_mse(pred, target, train_mask)
            eval_mse = _masked_mse(pred, target, ~train_mask)
            all_mse = jnp.mean((pred - target) ** 2)
            return train_mse, eval_mse, all_mse

        ct, ce, ca = _both(pred_call, self.call_prices, masks.call_train)
        pt, pe, pa = _both(pred_put, self.put_prices, masks.put_train)
        ot, oe, oa = _both(pred_otm, self.otm_put_prices, masks.otm_put_train)
        oct, oce, oca = _both(pred_otm_call, self.otm_call_prices, masks.otm_call_train)
        dt, de, da = _both(pred_digital, self.digital_put_prices, masks.digital_train)

        w = self.w_call + self.w_put + self.w_otm + self.w_otm_call + self.w_digital
        train_loss = (self.w_call * ct + self.w_put * pt + self.w_otm * ot
                      + self.w_otm_call * oct + self.w_digital * dt) / w
        eval_loss = (self.w_call * ce + self.w_put * pe + self.w_otm * oe
                     + self.w_otm_call * oce + self.w_digital * de) / w
        all_loss = (self.w_call * ca + self.w_put * pa + self.w_otm * oa
                    + self.w_otm_call * oca + self.w_digital * da) / w

        metrics = {
            "train_loss": train_loss, "eval_loss": eval_loss, "all_loss": all_loss,
            "train_call": ct, "eval_call": ce,
            "train_put": pt, "eval_put": pe,
            "train_otm": ot, "eval_otm": oe,
            "train_otm_call": oct, "eval_otm_call": oce,
            "train_digital": dt, "eval_digital": de,
            "pred_call": pred_call, "pred_put": pred_put,
            "pred_otm": pred_otm, "pred_otm_call": pred_otm_call,
            "pred_digital": pred_digital,
        }
        return metrics

    def train(self, seed: int) -> dict:
        tc = self.train_config

        run = None
        if tc.use_wandb:
            import wandb
            run = wandb.init(
                project=tc.wandb_project,
                config={"seed": seed, "config_name": self.config_name},
                name=f"dax2_vanilla_{self.config_name}_seed{seed}",
                group=f"dax2_vanilla_{self.config_name}", reinit=True,
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

        trainer = self
        train_bs = tc.batch_size
        eval_bs = tc.eval_batch_size

        @nnx.jit
        def train_step(model, optimizer):
            def loss_fn(model):
                return trainer._compute_train_loss(model, train_bs)
            (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
            optimizer.update(model, grads)
            grad_norm = optax.global_norm(grads)
            return loss, metrics, grad_norm

        @nnx.jit
        def eval_step(model):
            return trainer._compute_eval_metrics(model, eval_bs)

        history = {
            "train_loss": [], "eval_loss": [], "all_loss": [],
            "lr": [], "grad_norm": [],
            "train_call": [], "eval_call": [],
            "train_otm": [], "eval_otm": [],
            "train_digital": [], "eval_digital": [],
        }

        pbar = tqdm(range(tc.num_epochs), desc=f"dax2_vanilla {self.config_name} seed={seed}")
        t_start = time.time()

        for step in pbar:
            loss, metrics, grad_norm = train_step(model, optimizer)
            lr = schedule(step)
            loss_val = float(loss)
            history["train_loss"].append(loss_val)
            history["lr"].append(float(lr))
            history["grad_norm"].append(float(grad_norm))

            if step % tc.log_every == 0 or step == tc.num_epochs - 1:
                pbar.set_postfix(loss=f"{loss_val:.6f}", lr=f"{float(lr):.2e}")

            if step % tc.eval_every == 0 or step == tc.num_epochs - 1:
                em = eval_step(model)
                history["eval_loss"].append(float(em["eval_loss"]))
                history["all_loss"].append(float(em["all_loss"]))
                history["train_call"].append(float(em["train_call"]))
                history["eval_call"].append(float(em["eval_call"]))
                history["train_otm"].append(float(em["train_otm"]))
                history["eval_otm"].append(float(em["eval_otm"]))
                history["train_digital"].append(float(em["train_digital"]))
                history["eval_digital"].append(float(em["eval_digital"]))

        elapsed = time.time() - t_start
        print(f"\ndax2_vanilla {self.config_name} seed={seed} done in {elapsed:.1f}s")

        em = eval_step(model)
        _, params, _ = nnx.split(model, nnx.Param, nnx.RngState)

        final_result = {
            "final_train_loss": float(em["train_loss"]),
            "final_eval_loss": float(em["eval_loss"]),
            "final_all_loss": float(em["all_loss"]),
            "final_overfit_gap": float(em["eval_loss"] - em["train_loss"]),
            # Per-product MSEs (train mask, eval mask).  Mirrors the keys
            # exposed by Dax3GirsanovTrainer so vanilla/girsanov result
            # CSVs are directly comparable column-by-column.
            "final_train_call": float(em["train_call"]),
            "final_eval_call": float(em["eval_call"]),
            "final_train_put": float(em["train_put"]),
            "final_eval_put": float(em["eval_put"]),
            "final_train_otm": float(em["train_otm"]),
            "final_eval_otm": float(em["eval_otm"]),
            "final_train_otm_call": float(em["train_otm_call"]),
            "final_eval_otm_call": float(em["eval_otm_call"]),
            "final_train_digital": float(em["train_digital"]),
            "final_eval_digital": float(em["eval_digital"]),
            "final_mae_call_train": float(jnp.mean(jnp.abs(
                (em["pred_call"] - self.call_prices) * self.masks.call_train[:, None]))),
            "final_mae_call_eval": float(jnp.mean(jnp.abs(
                (em["pred_call"] - self.call_prices) * self.masks.call_eval[:, None]))),
        }

        if tc.use_wandb and run is not None:
            import wandb
            wandb.summary.update(final_result)
            wandb.finish()

        return {"params": params, "history": history, "final_metrics": final_result,
                "config_name": self.config_name, "seed": seed}

    def generate_paths(self, params, n_paths: int = 50, seed: int = 999):
        rngs = nnx.Rngs(params=0, init=seed, noise=seed + 1)
        model = _build_model(self.model_config, rngs, model_cls=self.model_cls)
        graphdef, _, rng_state = nnx.split(model, nnx.Param, nnx.RngState)
        model = nnx.merge(graphdef, params, rng_state)
        return model(batch_size=n_paths, return_path=True, y0=self.y0)[:, :, 0]
