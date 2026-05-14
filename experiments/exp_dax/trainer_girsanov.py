# v2/experiments/exp_dax/trainer_girsanov.py
"""
DAX4 trainer with TWO last-layer Girsanov tilts (one per OTM side).

Mirrors :class:`Dax3GirsanovTrainer` but uses ``cfg.num_girsanov = 2``:

  * Controller 0  →  OTM-put  SNIS estimator
  * Controller 1  →  OTM-call SNIS estimator

ATM call/put losses always stay under P (they're insensitive to rare-
event reweighting).  The digital-put loss stays under P too — the OTM-
put controller is optimised for the put barrier itself, but using its
weights for the digital indicator can introduce extra variance, so we
keep digitals under P unless the user explicitly opts in.

The model returns

    (output_p, [output_q_put, output_q_call], [log_rn_put, log_rn_call])

so the trainer runs two SNIS estimates on disjoint Q-batches.
"""
import time

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from tqdm import tqdm

from v3.benchmarks.neural_sde import NeuralSDEStack
from v3.benchmarks.neural_sde_config import NeuralSDEConfig
from v3.benchmarks.slice_model import SLiCEStack
from v3.benchmarks.slice_config import SLiCEConfig
from v3.experiments.configs import TrainConfig
from v3.src.models.model import SLiSDEModel
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
    diff2 = (pred - target) ** 2
    m = mask[:, None].astype(diff2.dtype)
    return jnp.sum(diff2 * m) / (jnp.sum(m) * target.shape[1] + 1e-12)


class GirsanovTrainer:
    """DAX4 trainer with two Girsanov controllers (one per OTM side)."""

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

        self.is_neural_sde = isinstance(model_config, NeuralSDEConfig)
        self.use_girsanov = (
            (not self.is_neural_sde)
            and getattr(model_config, "use_girsanov", False)
        )
        # v3 dax girsanov uses exactly two controllers (one for OTM-put SNIS,
        # one for OTM-call SNIS).  Pin ``num_girsanov`` on the config so a
        # mis-specified script does not silently fall back to a single
        # shared controller.
        if self.use_girsanov:
            if int(getattr(model_config, "num_girsanov", 1)) != 2:
                model_config.num_girsanov = 2
            self.n_girsanov = 2
        else:
            self.n_girsanov = 1

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

    # ------------------------------------------------------------------
    # Prediction helpers (mirrors dax3, with separate Q-paths per side)
    # ------------------------------------------------------------------

    def _atm_predictions_p(self, z_p):
        Y_at_mats = z_p[:, self.mat_indices]
        pred_call = jnp.mean(
            jax.nn.relu(Y_at_mats[:, None, :] - self.call_strikes[None, :, None]),
            axis=0,
        )
        pred_put = jnp.mean(
            jax.nn.relu(self.put_strikes[None, :, None] - Y_at_mats[:, None, :]),
            axis=0,
        )
        return pred_call, pred_put

    def _digital_p(self, z_p):
        Y_at_mats = z_p[:, self.mat_indices]
        sharpness = 50.0
        digital_indicator = jax.nn.sigmoid(
            sharpness * (self.digital_put_strikes[None, :, None] - Y_at_mats[:, None, :])
        )
        return jnp.mean(digital_indicator, axis=0)

    def _otm_put_p(self, z_p):
        Y_at_mats = z_p[:, self.mat_indices]
        return jnp.mean(
            jax.nn.relu(self.otm_put_strikes[None, :, None] - Y_at_mats[:, None, :]),
            axis=0,
        )

    def _otm_call_p(self, z_p):
        Y_at_mats = z_p[:, self.mat_indices]
        return jnp.mean(
            jax.nn.relu(Y_at_mats[:, None, :] - self.otm_call_strikes[None, :, None]),
            axis=0,
        )

    def _snis_weights(self, log_rn):
        """Stabilised SNIS weights.  Same recipe as dax3."""
        tc = self.train_config
        log_w = -log_rn
        if tc.log_weight_clip > 0:
            c = tc.log_weight_clip
            log_w = jnp.clip(log_w, -c, c)
        log_w = log_w - jax.lax.stop_gradient(jnp.max(log_w))
        w = jnp.exp(log_w)
        if tc.weight_clip > 0:
            w = jnp.clip(w, 0.0, tc.weight_clip)
        w_norm = w / (jnp.sum(w) + 1e-12)
        ess = (jnp.sum(w) ** 2) / (jnp.sum(w ** 2) + 1e-12)
        return w_norm, ess

    def _otm_put_q(self, z_q_put, log_rn_put):
        Y_at_mats = z_q_put[:, self.mat_indices]
        otm_put = jax.nn.relu(
            self.otm_put_strikes[None, :, None] - Y_at_mats[:, None, :]
        )
        w_norm, ess = self._snis_weights(log_rn_put)
        pred = jnp.einsum("b,bij->ij", w_norm, otm_put)
        return pred, ess

    def _otm_call_q(self, z_q_call, log_rn_call):
        Y_at_mats = z_q_call[:, self.mat_indices]
        otm_call = jax.nn.relu(
            Y_at_mats[:, None, :] - self.otm_call_strikes[None, :, None]
        )
        w_norm, ess = self._snis_weights(log_rn_call)
        pred = jnp.einsum("b,bij->ij", w_norm, otm_call)
        return pred, ess

    # ------------------------------------------------------------------
    # Loss assembly
    # ------------------------------------------------------------------

    def _assemble(self, pred_call, pred_put, pred_otm, pred_otm_call, pred_digital):
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
            "pred_call": pred_call, "pred_put": pred_put,
            "pred_otm": pred_otm, "pred_otm_call": pred_otm_call,
            "pred_digital": pred_digital,
        }
        return loss, metrics

    def _compute_loss_p(self, model, batch_size):
        clip_z = self.train_config.clip_z
        paths = model(
            batch_size=batch_size, return_path=True, y0=self.y0,
            force_no_girsanov=True,
        )
        z_p = jnp.clip(paths[:, :, 0], -clip_z, clip_z)
        pred_call, pred_put = self._atm_predictions_p(z_p)
        pred_otm = self._otm_put_p(z_p)
        pred_otm_call = self._otm_call_p(z_p)
        pred_digital = self._digital_p(z_p)
        loss, metrics = self._assemble(
            pred_call, pred_put, pred_otm, pred_otm_call, pred_digital
        )
        metrics["ess_put"] = jnp.array(float(batch_size))
        metrics["ess_call"] = jnp.array(float(batch_size))
        metrics["kl_qp_put"] = jnp.array(0.0)
        metrics["kl_qp_call"] = jnp.array(0.0)
        return loss, metrics

    def _compute_loss_q(self, model, batch_size_p, batch_size_q):
        tc = self.train_config
        clip_z = tc.clip_z
        # Multi-tilt forward: returns lists indexed by controller.
        paths_p, paths_q_list, log_rn_list = model(
            batch_size=batch_size_p, batch_size_q=batch_size_q,
            return_path=True, y0=self.y0,
        )
        z_p = jnp.clip(paths_p[:, :, 0], -clip_z, clip_z)
        z_q_put = jnp.clip(paths_q_list[0][:, :, 0], -clip_z, clip_z)
        z_q_call = jnp.clip(paths_q_list[1][:, :, 0], -clip_z, clip_z)
        log_rn_put, log_rn_call = log_rn_list[0], log_rn_list[1]

        pred_call, pred_put = self._atm_predictions_p(z_p)
        pred_digital = self._digital_p(z_p)
        pred_otm, ess_put = self._otm_put_q(z_q_put, log_rn_put)
        pred_otm_call, ess_call = self._otm_call_q(z_q_call, log_rn_call)

        loss_mse, metrics = self._assemble(
            pred_call, pred_put, pred_otm, pred_otm_call, pred_digital
        )
        metrics["ess_put"] = ess_put
        metrics["ess_call"] = ess_call
        kl_qp_put = jnp.mean(log_rn_put)
        kl_qp_call = jnp.mean(log_rn_call)
        metrics["kl_qp_put"] = kl_qp_put
        metrics["kl_qp_call"] = kl_qp_call
        # ``loss_mse`` is the MSE-only loss reported in metrics so that
        # Girsanov on/off runs are directly comparable.  KL anchors are
        # added only to the gradient target ``loss``.
        loss = loss_mse
        if tc.kl_weight > 0:
            loss = loss + tc.kl_weight * (kl_qp_put + kl_qp_call)
        return loss, metrics

    # Eval — mask-aware per-product split.
    def _compute_eval(self, model, batch_size, girsanov_active):
        clip_z = self.train_config.clip_z
        if girsanov_active:
            paths_p, paths_q_list, log_rn_list = model(
                batch_size=batch_size, batch_size_q=batch_size,
                return_path=True, y0=self.y0,
            )
            z_p = jnp.clip(paths_p[:, :, 0], -clip_z, clip_z)
            z_q_put = jnp.clip(paths_q_list[0][:, :, 0], -clip_z, clip_z)
            z_q_call = jnp.clip(paths_q_list[1][:, :, 0], -clip_z, clip_z)
            log_rn_put, log_rn_call = log_rn_list[0], log_rn_list[1]
            pred_call, pred_put = self._atm_predictions_p(z_p)
            pred_digital = self._digital_p(z_p)
            pred_otm, ess_put = self._otm_put_q(z_q_put, log_rn_put)
            pred_otm_call, ess_call = self._otm_call_q(z_q_call, log_rn_call)
            kl_qp_put = jnp.mean(log_rn_put)
            kl_qp_call = jnp.mean(log_rn_call)
        else:
            paths = model(
                batch_size=batch_size, return_path=True, y0=self.y0,
                force_no_girsanov=True,
            )
            z_p = jnp.clip(paths[:, :, 0], -clip_z, clip_z)
            pred_call, pred_put = self._atm_predictions_p(z_p)
            pred_otm = self._otm_put_p(z_p)
            pred_otm_call = self._otm_call_p(z_p)
            pred_digital = self._digital_p(z_p)
            ess_put = jnp.array(float(batch_size))
            ess_call = jnp.array(float(batch_size))
            kl_qp_put = jnp.array(0.0)
            kl_qp_call = jnp.array(0.0)

        masks = self.masks

        def _both(pred, target, train_mask):
            train_mse = _masked_mse(pred, target, train_mask)
            eval_mse = _masked_mse(pred, target, ~train_mask)
            all_mse = jnp.mean((pred - target) ** 2)
            return train_mse, eval_mse, all_mse

        ct, ce, ca = _both(pred_call, self.call_prices, masks.call_train)
        pt, pe, pa = _both(pred_put, self.put_prices, masks.put_train)
        ot, oe, oa = _both(pred_otm, self.otm_put_prices, masks.otm_put_train)
        oct_, oce, oca = _both(pred_otm_call, self.otm_call_prices, masks.otm_call_train)
        dt, de, da = _both(pred_digital, self.digital_put_prices, masks.digital_train)

        w = self.w_call + self.w_put + self.w_otm + self.w_otm_call + self.w_digital
        train_loss = (self.w_call * ct + self.w_put * pt + self.w_otm * ot
                      + self.w_otm_call * oct_ + self.w_digital * dt) / w
        eval_loss = (self.w_call * ce + self.w_put * pe + self.w_otm * oe
                     + self.w_otm_call * oce + self.w_digital * de) / w
        all_loss = (self.w_call * ca + self.w_put * pa + self.w_otm * oa
                    + self.w_otm_call * oca + self.w_digital * da) / w

        return {
            "train_loss": train_loss, "eval_loss": eval_loss, "all_loss": all_loss,
            "train_call": ct, "eval_call": ce,
            "train_put": pt, "eval_put": pe,
            "train_otm": ot, "eval_otm": oe,
            "train_otm_call": oct_, "eval_otm_call": oce,
            "train_digital": dt, "eval_digital": de,
            "all_call": ca, "all_put": pa, "all_otm": oa,
            "all_otm_call": oca, "all_digital": da,
            "pred_call": pred_call, "pred_put": pred_put,
            "pred_otm": pred_otm, "pred_otm_call": pred_otm_call,
            "pred_digital": pred_digital,
            "ess_put": ess_put, "ess_call": ess_call,
            "kl_qp_put": kl_qp_put, "kl_qp_call": kl_qp_call,
        }

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, seed: int) -> dict:
        tc = self.train_config

        run = None
        if tc.use_wandb:
            import wandb
            run = wandb.init(
                project=tc.wandb_project,
                config={"seed": seed, "config_name": self.config_name,
                        "use_girsanov": self.use_girsanov,
                        "num_girsanov": self.n_girsanov},
                name=f"dax4_gir_{self.config_name}_seed{seed}",
                group=f"dax4_gir_{self.config_name}", reinit=True,
            )

        rngs = nnx.Rngs(params=seed, init=seed + 1000, noise=seed + 2000)
        model = _build_model(self.model_config, rngs, model_cls=self.model_cls)

        warmup_steps = min(tc.warmup_steps, tc.num_epochs // 2)
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=tc.end_lr, peak_value=tc.peak_lr,
            warmup_steps=warmup_steps, decay_steps=tc.num_epochs, end_value=tc.end_lr,
        )
        # ------------------------------------------------------------------
        # Two-optimiser setup so Girsanov only affects controller params.
        # Substring match (``exact=False``) is required here so the filter
        # catches ``tilt_controller_0`` and ``tilt_controller_1`` together —
        # the v2 model lays out multi-tilt controllers as separate
        # attributes rather than under a shared parent.
        # ------------------------------------------------------------------
        backbone_filter = nnx.All(
            nnx.Param,
            nnx.Not(nnx.PathContains("tilt_controller", exact=False)),
        )
        ctrl_filter = nnx.All(
            nnx.Param, nnx.PathContains("tilt_controller", exact=False)
        )
        backbone_diff = nnx.DiffState(0, backbone_filter)
        ctrl_diff = nnx.DiffState(0, ctrl_filter)

        opt_backbone_chain = optax.chain(
            optax.clip_by_global_norm(tc.grad_clip_norm),
            optax.adamw(learning_rate=schedule, weight_decay=tc.weight_decay),
        )
        opt_backbone = nnx.Optimizer(model, opt_backbone_chain, wrt=backbone_filter)

        if self.use_girsanov:
            opt_controller_chain = optax.chain(
                optax.clip_by_global_norm(tc.grad_clip_norm),
                optax.adamw(learning_rate=schedule, weight_decay=tc.weight_decay),
            )
            opt_controller = nnx.Optimizer(model, opt_controller_chain, wrt=ctrl_filter)
        else:
            opt_controller = None

        trainer = self
        train_bs = tc.batch_size
        eval_bs = tc.eval_batch_size
        bs_q = tc.girsanov_batch_size_q

        skip_nf = bool(getattr(tc, "skip_nonfinite_updates", True))

        def _safe_grads(grads, loss):
            grad_norm = optax.global_norm(grads)
            finite = jnp.isfinite(loss) & jnp.isfinite(grad_norm)
            if skip_nf:
                grads = jax.tree_util.tree_map(
                    lambda g: jnp.where(finite, g, jnp.zeros_like(g)),
                    grads,
                )
            return grads, grad_norm, finite

        @nnx.jit
        def train_step_p(model, opt_b):
            def loss_fn(model):
                return trainer._compute_loss_p(model, train_bs)
            (loss, metrics), grads = nnx.value_and_grad(
                loss_fn, has_aux=True, argnums=backbone_diff
            )(model)
            grads, grad_norm, finite = _safe_grads(grads, loss)
            opt_b.update(model, grads)
            metrics = {**metrics, "step_finite": finite.astype(jnp.float32)}
            return loss, metrics, grad_norm

        @nnx.jit
        def train_step_q(model, opt_b, opt_c):
            """Two-headed step: backbone <- L_p, controllers <- L_q.

            One controller-side update covers both tilt controllers:
            ``ctrl_filter`` includes ``tilt_controller_0`` *and*
            ``tilt_controller_1`` so a single ``argnums=ctrl_diff``
            gradient call returns gradients for both.
            """
            def loss_p_fn(model):
                return trainer._compute_loss_p(model, train_bs)
            (loss_p, _), grads_b = nnx.value_and_grad(
                loss_p_fn, has_aux=True, argnums=backbone_diff
            )(model)
            grads_b, gnorm_b, finite_b = _safe_grads(grads_b, loss_p)
            opt_b.update(model, grads_b)

            def loss_q_fn(model):
                return trainer._compute_loss_q(model, train_bs, bs_q)
            (loss_q, metrics_q), grads_c = nnx.value_and_grad(
                loss_q_fn, has_aux=True, argnums=ctrl_diff
            )(model)
            grads_c, gnorm_c, finite_c = _safe_grads(grads_c, loss_q)
            opt_c.update(model, grads_c)

            metrics = {
                **metrics_q,
                "step_finite": (finite_b & finite_c).astype(jnp.float32),
                "loss_p_backbone": loss_p,
                "grad_norm_backbone": gnorm_b,
            }
            return loss_q, metrics, gnorm_c

        @nnx.jit
        def eval_step_p(model):
            return trainer._compute_eval(model, eval_bs, girsanov_active=False)

        @nnx.jit
        def eval_step_q(model):
            return trainer._compute_eval(model, eval_bs, girsanov_active=True)

        history = {
            "train_loss": [], "eval_loss": [], "all_loss": [],
            "lr": [], "grad_norm": [],
            "ess_put": [], "ess_call": [], "girsanov_active": [],
            "train_otm": [], "eval_otm": [],
            "train_otm_call": [], "eval_otm_call": [],
            "train_digital": [], "eval_digital": [],
            "step_finite": [],
        }

        pbar = tqdm(range(tc.num_epochs), desc=f"dax4_gir {self.config_name} seed={seed}")
        t_start = time.time()
        start_after = tc.start_girsanov_after_epoch

        for step in pbar:
            girsanov_now = self.use_girsanov and (step >= start_after)
            if girsanov_now:
                loss, metrics, grad_norm = train_step_q(
                    model, opt_backbone, opt_controller
                )
            else:
                loss, metrics, grad_norm = train_step_p(model, opt_backbone)
            lr = schedule(step)
            loss_val = float(loss)
            history["train_loss"].append(loss_val)
            history["lr"].append(float(lr))
            history["grad_norm"].append(float(grad_norm))
            history["girsanov_active"].append(int(girsanov_now))
            history["step_finite"].append(int(float(metrics["step_finite"])))

            if step % tc.log_every == 0 or step == tc.num_epochs - 1:
                postfix = dict(
                    loss=f"{loss_val:.4e}",
                    lr=f"{float(lr):.2e}",
                    G=("Q" if girsanov_now else "P"),
                    ess_p=f"{float(metrics['ess_put']):.1f}",
                    ess_c=f"{float(metrics['ess_call']):.1f}",
                )
                skipped = sum(1 for s in history["step_finite"] if s == 0)
                if skipped:
                    postfix["skip"] = str(skipped)
                pbar.set_postfix(**postfix)

            if step % tc.eval_every == 0 or step == tc.num_epochs - 1:
                em = eval_step_q(model) if girsanov_now else eval_step_p(model)
                history["eval_loss"].append(float(em["eval_loss"]))
                history["all_loss"].append(float(em["all_loss"]))
                history["train_otm"].append(float(em["train_otm"]))
                history["eval_otm"].append(float(em["eval_otm"]))
                history["train_otm_call"].append(float(em["train_otm_call"]))
                history["eval_otm_call"].append(float(em["eval_otm_call"]))
                history["train_digital"].append(float(em["train_digital"]))
                history["eval_digital"].append(float(em["eval_digital"]))
                history["ess_put"].append(float(em["ess_put"]))
                history["ess_call"].append(float(em["ess_call"]))

        elapsed = time.time() - t_start
        print(f"\ndax4_gir {self.config_name} seed={seed} done in {elapsed:.1f}s")

        em = eval_step_q(model) if self.use_girsanov else eval_step_p(model)
        _, params, _ = nnx.split(model, nnx.Param, nnx.RngState)

        final_result = {
            "final_train_loss": float(em["train_loss"]),
            "final_eval_loss": float(em["eval_loss"]),
            "final_all_loss": float(em["all_loss"]),
            "final_overfit_gap": float(em["eval_loss"] - em["train_loss"]),
            "final_ess_put": float(em["ess_put"]),
            "final_ess_call": float(em["ess_call"]),
            "final_kl_qp_put": float(em.get("kl_qp_put", 0.0)),
            "final_kl_qp_call": float(em.get("kl_qp_call", 0.0)),
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
            "final_all_call": float(em["all_call"]),
            "final_all_put": float(em["all_put"]),
            "final_all_otm": float(em["all_otm"]),
            "final_all_otm_call": float(em["all_otm_call"]),
            "final_all_digital": float(em["all_digital"]),
            "use_girsanov": bool(self.use_girsanov),
            "num_girsanov": int(self.n_girsanov),
        }

        if tc.use_wandb and run is not None:
            import wandb
            wandb.summary.update(final_result)
            wandb.finish()

        return {"params": params, "history": history, "final_metrics": final_result,
                "config_name": self.config_name, "seed": seed}

    def generate_paths(self, params, n_paths: int = 50, seed: int = 999,
                       use_girsanov: bool = False):
        rngs = nnx.Rngs(params=0, init=seed, noise=seed + 1)
        model = _build_model(self.model_config, rngs, model_cls=self.model_cls)
        graphdef, _, rng_state = nnx.split(model, nnx.Param, nnx.RngState)
        model = nnx.merge(graphdef, params, rng_state)
        # Always sample under P for path visualisation; the multi-tilt
        # all-Q mode is rejected by the model anyway.
        paths = model(batch_size=n_paths, return_path=True, y0=self.y0,
                      force_no_girsanov=True)
        return paths[:, :, 0]
