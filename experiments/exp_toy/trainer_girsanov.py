# v2/experiments/exp_toy/trainer_exp2.py
"""
Trainer for exp_toy with last-layer-only Girsanov tilt support.

The trainer mirrors :class:`Exp1Trainer` but adds a Girsanov branch
controlled by ``model_config.use_girsanov``:

* During the warm-up phase (``epoch < start_girsanov_after_epoch``) the
  forward pass is invoked with ``force_no_girsanov=True`` so all four
  loss components are estimated under P, exactly as in Exp1.

* Once the warm-up ends, every step runs a *dual-batch shared-prefix*
  forward: ``batch_size`` paths under P (used for payoff/runmax/sqavg)
  plus ``girsanov_batch_size_q`` paths under Q (used for the barrier
  loss via a self-normalised importance-sampling estimator).

The barrier estimator under Q is:

    w_i      = exp(-log_rn_i)                 # dP/dQ for path i
    pred     = sum_i w_i 1{path_i hit barrier} / sum_i w_i

We use the self-normalised form because (a) the unnormalised IS
estimator has very high variance early in training (when the tilt is
still random) and (b) the warm-started ``L_mix`` initialisation already
guarantees a small bias.
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
from v3.datasets import CalibrationDataset


def _build_model(cfg, rngs, model_cls=None):
    """Build the model matching ``cfg``; only SLiSDE has a Girsanov branch."""
    if model_cls is not None:
        return model_cls(config=cfg, rngs=rngs)
    if isinstance(cfg, NeuralSDEConfig):
        return NeuralSDEStack(config=cfg, rngs=rngs)
    if isinstance(cfg, SLiCEConfig):
        return SLiCEStack(config=cfg, rngs=rngs)
    return SLiSDEModel(config=cfg, rngs=rngs)


class GirsanovTrainer:
    """exp_toy trainer with optional last-layer-only Girsanov tilt.

    Loss components (always 4):
        - L_payoff   : European-call MSE
        - L_runmax   : running-maximum MSE
        - L_sqavg    : squared-average MSE
        - L_barrier  : barrier-hit-probability MSE

    During warm-up all four are computed under P.  After warm-up,
    L_payoff / L_runmax / L_sqavg stay under P and L_barrier switches
    to a self-normalised IS estimator under Q.
    """

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
        self.model_cls = model_cls
        self.is_neural_sde = isinstance(model_config, NeuralSDEConfig)

        # Girsanov is only meaningful for SLiSDE — Neural-SDE benchmarks
        # don't expose a tilt.  Force it off for them.
        self.use_girsanov = (
            (not self.is_neural_sde)
            and getattr(model_config, "use_girsanov", False)
        )

        self.target_payoffs = dataset.target_payoffs
        self.target_running_max = dataset.target_running_max
        self.target_squared_avg = dataset.target_squared_avg
        self.target_barrier_probs = dataset.target_barrier_probs
        self.barriers = dataset.barriers
        self.strikes = dataset.strikes
        self.mat_indices = dataset.maturity_indices
        self.n_steps = model_config.n_steps
        self.y0 = dataset.y0

    # ------------------------------------------------------------------
    # Loss helpers
    # ------------------------------------------------------------------

    def _standard_losses_from_z(self, z):
        """Compute payoff / runmax / sqavg / barrier from z under P (uniform weights)."""
        n_steps = self.n_steps
        z_at_mats = z[:, self.mat_indices]
        payoffs = jax.nn.relu(z_at_mats[:, None, :] - self.strikes[None, :, None])
        pred_payoffs = jnp.mean(payoffs, axis=0)

        cummax = jax.lax.associative_scan(jnp.maximum, z, axis=1)
        pred_runmax = jnp.mean(cummax[:, self.mat_indices], axis=0)

        cumsq = jnp.cumsum(z ** 2, axis=1)
        divisors = jnp.arange(1, n_steps + 2, dtype=z.dtype)[None, :]
        pred_sqavg = jnp.mean((cumsq / divisors)[:, self.mat_indices], axis=0)

        runmax_at_mats = cummax[:, self.mat_indices]
        hit = (runmax_at_mats[:, None, :] >= self.barriers[None, :, None]).astype(z.dtype)
        pred_barrier = jnp.mean(hit, axis=0)
        return pred_payoffs, pred_runmax, pred_sqavg, pred_barrier

    def _barrier_under_q(self, z_q, log_rn_q):
        """Self-normalised IS estimator of the barrier-hit probability.

        Stabilisation pipeline (in order):
        1.  Symmetric clip on ``log w`` if ``log_weight_clip > 0`` — caps
            the dynamic range of the weights so a single outlier path
            cannot dominate the SNIS sum.
        2.  Max-shift  ``log w -= max(log w)``  (no-op for the SNIS mean,
            keeps ``exp`` away from overflow).  We stop-gradient through
            the max so AdamW doesn't see a step-discontinuous gradient
            when the argmax index swaps between batches.
        3.  Optional hard clip on ``w`` if ``weight_clip > 0``.

        Backbone-vs-controller gradient separation is handled in
        ``SLiSDEModel.__call__`` by ``stop_gradient`` on the controller's
        ``Z^{(L-1)}`` input (the only path through which SNIS gradients
        could leak into the SLiSDE backbone).  Here we keep the
        controller's gradient through ``log_rn_q`` intact.
        """
        tc = self.train_config
        cummax = jax.lax.associative_scan(jnp.maximum, z_q, axis=1)
        runmax_at_mats = cummax[:, self.mat_indices]
        hit = (runmax_at_mats[:, None, :] >= self.barriers[None, :, None]).astype(z_q.dtype)

        # log w_i = -log_rn_i  (dP/dQ).
        log_w = -log_rn_q
        if tc.log_weight_clip > 0:
            c = tc.log_weight_clip
            log_w = jnp.clip(log_w, -c, c)
        log_w = log_w - jax.lax.stop_gradient(jnp.max(log_w))
        w = jnp.exp(log_w)
        if tc.weight_clip > 0:
            w = jnp.clip(w, 0.0, tc.weight_clip)

        w_norm = w / (jnp.sum(w) + 1e-12)
        pred_barrier_q = jnp.einsum("b,bij->ij", w_norm, hit)

        ess = (jnp.sum(w) ** 2) / (jnp.sum(w ** 2) + 1e-12)
        return pred_barrier_q, ess

    def _assemble_loss(self, pred_payoffs, pred_runmax, pred_sqavg, pred_barrier):
        bw = self.train_config.barrier_weight
        loss_payoffs = jnp.mean((pred_payoffs - self.target_payoffs) ** 2)
        loss_runmax = jnp.mean((pred_runmax - self.target_running_max) ** 2)
        loss_sqavg = jnp.mean((pred_sqavg - self.target_squared_avg) ** 2)
        loss_barrier = jnp.mean((pred_barrier - self.target_barrier_probs) ** 2)
        loss_std = (loss_payoffs + loss_runmax + loss_sqavg) / 3.0
        loss = (loss_std + bw * loss_barrier) / (1.0 + bw)
        return loss, loss_payoffs, loss_runmax, loss_sqavg, loss_barrier

    def _compute_loss_p(self, model, batch_size):
        """All four losses estimated under P (warm-up / no Girsanov)."""
        clip_z = self.train_config.clip_z
        paths = model(
            batch_size=batch_size, return_path=True, y0=self.y0,
            force_no_girsanov=True,
        )
        z = jnp.clip(paths[:, :, 0], -clip_z, clip_z)
        pp, pr, ps, pb = self._standard_losses_from_z(z)
        loss, lp, lr, lsq, lbar = self._assemble_loss(pp, pr, ps, pb)
        metrics = {
            "loss": loss, "loss_payoffs": lp, "loss_runmax": lr,
            "loss_sqavg": lsq, "loss_barrier": lbar,
            "pred_payoffs": pp, "pred_runmax": pr,
            "pred_sqavg": ps, "pred_barrier": pb,
            "ess": jnp.array(float(batch_size)),
            "kl_qp": jnp.array(0.0),
        }
        return loss, metrics

    def _compute_loss_q(self, model, batch_size_p, batch_size_q):
        """Dual-batch loss: P-paths for std losses, Q-paths for barrier."""
        tc = self.train_config
        clip_z = tc.clip_z
        paths_p, paths_q, log_rn_q = model(
            batch_size=batch_size_p,
            batch_size_q=batch_size_q,
            return_path=True, y0=self.y0,
        )
        z_p = jnp.clip(paths_p[:, :, 0], -clip_z, clip_z)
        z_q = jnp.clip(paths_q[:, :, 0], -clip_z, clip_z)

        pp, pr, ps, _ = self._standard_losses_from_z(z_p)
        pred_barrier_q, ess = self._barrier_under_q(z_q, log_rn_q)

        loss_mse, lp, lr, lsq, lbar = self._assemble_loss(pp, pr, ps, pred_barrier_q)

        # ``loss_mse`` is the comparable MSE-only loss (same definition as
        # ``_compute_loss_p``) and is what we report so that runs with /
        # without Girsanov can be compared directly.  The KL(Q‖P) anchor —
        # E_Q[log_rn] under SNIS — is added only to the *gradient target*
        # ``loss``, never to the reported metric.
        kl_qp = jnp.mean(log_rn_q)
        loss = loss_mse + tc.kl_weight * kl_qp if tc.kl_weight > 0 else loss_mse

        metrics = {
            "loss": loss_mse, "loss_payoffs": lp, "loss_runmax": lr,
            "loss_sqavg": lsq, "loss_barrier": lbar,
            "pred_payoffs": pp, "pred_runmax": pr,
            "pred_sqavg": ps, "pred_barrier": pred_barrier_q,
            "ess": ess, "kl_qp": kl_qp,
        }
        return loss, metrics

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
                        "use_girsanov": self.use_girsanov},
                name=f"girsanov_{self.config_name}_seed{seed}",
                group=f"girsanov_{self.config_name}", reinit=True,
            )

        rngs = nnx.Rngs(params=seed, init=seed + 1000, noise=seed + 2000)
        model = _build_model(self.model_config, rngs, model_cls=self.model_cls)

        warmup_steps = min(tc.warmup_steps, tc.num_epochs // 2)
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=tc.end_lr, peak_value=tc.peak_lr,
            warmup_steps=warmup_steps, decay_steps=tc.num_epochs, end_value=tc.end_lr,
        )
        # ------------------------------------------------------------------
        # Two-optimiser setup so Girsanov only affects the *controller*.
        #
        # Without this split, gradients of L_barrier (computed under Q via
        # SNIS) flow back into the SLiSDE backbone through the *shared*
        # last-layer parameters and the shared prefix.  When ESS is low
        # (the typical SNIS regime), those gradients are extremely noisy
        # and corrupt the backbone — destroying the P-side
        # payoff/runmax/sqavg predictions even though they are estimated
        # under P.  By giving the backbone and the controller their own
        # AdamW state filtered by NNX path, the barrier loss can never
        # touch the backbone, and the std losses can never touch the
        # controller.  Strict separation matches the role of each block:
        # backbone = SDE drift, controller = IS proposal.
        # ------------------------------------------------------------------
        backbone_filter = nnx.All(
            nnx.Param, nnx.Not(nnx.PathContains("tilt_controller"))
        )
        ctrl_filter = nnx.All(nnx.Param, nnx.PathContains("tilt_controller"))
        # NNX uses ``DiffState(argnum, filter)`` to take grads wrt a subset
        # of params.  ``argnums=`` accepts either an int or a DiffState.
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
            """Sanitise grads: zero them out when loss / grad-norm is non-finite.

            Without this a single SNIS blow-up corrupts AdamW's moments and
            every subsequent update — including the pure-P backbone update —
            also returns NaN.  We compute the global grad-norm once, mask the
            whole pytree when the step is unsafe, and AdamW takes a no-op.
            """
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
            """Two-headed Girsanov step.

            (a) Backbone <- grad_{backbone} L_p   (P-only forward, std losses)
            (b) Controller <- grad_{controller} L_q  (dual-batch, SNIS barrier
                loss, plus optional KL(Q||P) anchor when ``kl_weight > 0``)

            Because z_p does not depend on the controller and the SNIS
            gradient is restricted to controller params via ``wrt=`` filter,
            the two updates are gradient-disjoint by construction.
            """
            # (a) Backbone update from L_p — std losses under P only.
            def loss_p_fn(model):
                return trainer._compute_loss_p(model, train_bs)
            (loss_p, metrics_p), grads_b = nnx.value_and_grad(
                loss_p_fn, has_aux=True, argnums=backbone_diff
            )(model)
            grads_b, gnorm_b, finite_b = _safe_grads(grads_b, loss_p)
            opt_b.update(model, grads_b)

            # (b) Controller update from L_q — barrier loss under Q via SNIS
            #     (+ optional KL(Q||P) regulariser).  Std losses appear in
            #     L_q for reporting parity but contribute zero gradient to
            #     the controller, so the controller learns purely from the
            #     SNIS barrier estimator and the KL anchor.
            def loss_q_fn(model):
                return trainer._compute_loss_q(model, train_bs, bs_q)
            (loss_q, metrics_q), grads_c = nnx.value_and_grad(
                loss_q_fn, has_aux=True, argnums=ctrl_diff
            )(model)
            grads_c, gnorm_c, finite_c = _safe_grads(grads_c, loss_q)
            opt_c.update(model, grads_c)

            # Report the L_q metrics (headline barrier / ESS / KL) and the
            # controller grad-norm for monitoring SNIS health.
            metrics = {
                **metrics_q,
                "step_finite": (finite_b & finite_c).astype(jnp.float32),
                "loss_p_backbone": loss_p,
                "grad_norm_backbone": gnorm_b,
            }
            return loss_q, metrics, gnorm_c

        @nnx.jit
        def eval_step_p(model):
            return trainer._compute_loss_p(model, eval_bs)

        @nnx.jit
        def eval_step_q(model):
            return trainer._compute_loss_q(model, eval_bs, bs_q)

        history = {
            "train_loss": [], "eval_loss": [], "lr": [], "grad_norm": [],
            "loss_payoffs": [], "loss_runmax": [], "loss_sqavg": [],
            "loss_barrier": [], "ess": [], "girsanov_active": [],
            "step_finite": [], "kl_qp": [],
        }

        pbar = tqdm(range(tc.num_epochs), desc=f"exp2 {self.config_name} seed={seed}")
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
            history["kl_qp"].append(float(metrics.get("kl_qp", 0.0)))

            if step % tc.log_every == 0 or step == tc.num_epochs - 1:
                postfix = dict(
                    loss=f"{loss_val:.4e}",
                    bar=f"{float(metrics['loss_barrier']):.4e}",
                    lr=f"{float(lr):.2e}",
                )
                if girsanov_now:
                    postfix["ess"] = f"{float(metrics['ess']):.1f}"
                    postfix["G"] = "Q"
                else:
                    postfix["G"] = "P"
                # Number of steps so far where the optimiser update was
                # skipped because of a non-finite loss / grad — should stay
                # at 0 in healthy runs.
                skipped = sum(1 for s in history["step_finite"] if s == 0)
                if skipped:
                    postfix["skip"] = str(skipped)
                pbar.set_postfix(**postfix)
                if tc.use_wandb and run is not None:
                    import wandb
                    wandb.log({
                        "train/loss": loss_val,
                        "train/loss_barrier": float(metrics["loss_barrier"]),
                        "train/ess": float(metrics["ess"]),
                        "train/girsanov_active": int(girsanov_now),
                        "step": step,
                    })

            if step % tc.eval_every == 0 or step == tc.num_epochs - 1:
                if girsanov_now:
                    eval_loss, eval_metrics = eval_step_q(model)
                else:
                    eval_loss, eval_metrics = eval_step_p(model)
                history["eval_loss"].append(float(eval_loss))
                history["loss_payoffs"].append(float(eval_metrics["loss_payoffs"]))
                history["loss_runmax"].append(float(eval_metrics["loss_runmax"]))
                history["loss_sqavg"].append(float(eval_metrics["loss_sqavg"]))
                history["loss_barrier"].append(float(eval_metrics["loss_barrier"]))
                history["ess"].append(float(eval_metrics["ess"]))

        elapsed = time.time() - t_start
        print(f"\nexp2 {self.config_name} seed={seed} done in {elapsed:.1f}s  "
              f"(final loss={history['train_loss'][-1]:.6f})")

        # Final eval — always under the final regime (Q if Girsanov is on).
        if self.use_girsanov:
            eval_loss, eval_metrics = eval_step_q(model)
        else:
            eval_loss, eval_metrics = eval_step_p(model)
        _, params, _ = nnx.split(model, nnx.Param, nnx.RngState)

        final_result = {
            # Report the MSE-only loss (no KL term) so Girsanov on/off runs
            # compare apples-to-apples; ``eval_loss`` is the grad target.
            "final_loss": float(eval_metrics["loss"]),
            "final_mae_payoffs": float(jnp.mean(jnp.abs(
                eval_metrics["pred_payoffs"] - self.target_payoffs))),
            "final_mae_runmax": float(jnp.mean(jnp.abs(
                eval_metrics["pred_runmax"] - self.target_running_max))),
            "final_mae_sqavg": float(jnp.mean(jnp.abs(
                eval_metrics["pred_sqavg"] - self.target_squared_avg))),
            "final_mae_barrier": float(jnp.mean(jnp.abs(
                eval_metrics["pred_barrier"] - self.target_barrier_probs))),
            "final_ess": float(eval_metrics["ess"]),
            "final_kl_qp": float(eval_metrics.get("kl_qp", 0.0)),
            "use_girsanov": bool(self.use_girsanov),
        }

        if tc.use_wandb and run is not None:
            import wandb
            wandb.summary.update({
                "final/loss": final_result["final_loss"],
                "final/ess": final_result["final_ess"],
                "elapsed_seconds": elapsed,
            })
            wandb.finish()

        return {
            "params": params,
            "history": history,
            "final_metrics": final_result,
            "config_name": self.config_name,
            "seed": seed,
        }

    def generate_paths(self, params, n_paths: int = 50, seed: int = 999,
                       use_girsanov: bool = False):
        rngs = nnx.Rngs(params=0, init=seed, noise=seed + 1)
        model = _build_model(self.model_config, rngs, model_cls=self.model_cls)
        graphdef, _, rng_state = nnx.split(model, nnx.Param, nnx.RngState)
        model = nnx.merge(graphdef, params, rng_state)
        if self.use_girsanov and use_girsanov:
            paths, _ = model(batch_size=n_paths, return_path=True, y0=self.y0)
        else:
            paths = model(batch_size=n_paths, return_path=True, y0=self.y0,
                          force_no_girsanov=True)
        return paths[:, :, 0]
