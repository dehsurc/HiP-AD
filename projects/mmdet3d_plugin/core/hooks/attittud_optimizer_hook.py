"""ATTITTUD Optimizer Hook for HiP-AD (planning-primary auxiliary gradient
decomposition), designed for per-sample (batch-1) training.

Adapts "Auxiliary Task Update Decomposition: The Good, The Bad and The
Neutral" (Dery et al., ICLR 2021) to HiP-AD multi-task training with
``plan`` as the primary task:

  1. Per-task backward passes expose each task's gradient on the *shared*
     parameter set (backbone / neck / unified-decoder shared ops).
  2. A low-rank subspace of recent planning gradients is maintained.
     - mode='svd_buffer': ring buffer of the last ``buffer_size`` planning
       gradients; Gram-matrix eigendecomposition gives the top-k basis where
       k is chosen adaptively as the smallest rank whose cumulative
       eigenvalue energy exceeds ``svd_energy`` (capped at ``k_max``).
     - mode='rank1': the current sample's planning gradient direction only.
  3. Every auxiliary gradient g_a is decomposed against that basis U and the
     current planning gradient g_p:
       - good:    in-subspace component whose coefficient agrees in sign
                  with g_p's coefficient           -> scaled by alpha_good
       - bad:     in-subspace component that disagrees in sign
                                                    -> scaled by alpha_bad
       - neutral: out-of-subspace (orthogonal) remainder
                                                    -> scaled by alpha_neutral
  4. Shared gradient is rebuilt as  g_p + sum_a g_a'  (+ untouched extra
     losses such as dense depth); task-private parameters keep their
     naturally accumulated per-task gradients.

The per-task loss plumbing reuses the PCGrad path: ``SparseDetector
.train_step`` (with ``_pcgrad_enabled``) exposes ``task_losses`` /
``aux_loss`` in ``runner.outputs``, and the shared-parameter grouping comes
from ``get_shared_parameters_grouped``.

Intended to run in fp32 (no Fp16OptimizerHook interplay) and with
``with_cp=False`` so multiple retain_graph backwards are safe.
"""

import logging
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from mmcv.runner import HOOKS, OptimizerHook

from .pcgrad_optimizer_hook import get_shared_parameters_grouped
from .nan_guard import skip_bad_step

logger = logging.getLogger(__name__)


@HOOKS.register_module()
class ATTITTUDOptimizerHook(OptimizerHook):
    """Planning-aware auxiliary gradient surgery (ATTITTUD variant).

    Args:
        shared_layers: names accepted by ``get_shared_parameters_grouped``
            (e.g. ['backbone', 'neck', 'inter_gnn', 'ffn', 'norm',
            'fc_before', 'fc_after']). Surgery operates on the single
            concatenated flat vector over all these groups.
        mode: 'svd_buffer' (ring buffer + adaptive-rank PCA) or 'rank1'
            (current planning gradient direction only).
        buffer_size: ring buffer length N for mode='svd_buffer'.
        svd_energy: cumulative eigenvalue-energy threshold in (0, 1] used to
            pick the basis rank k adaptively.
        k_max: hard cap on the basis rank.
        alpha_good / alpha_bad / alpha_neutral: scaling of the three
            components of each auxiliary gradient. ATTITTUD defaults would be
            (1, 0, 1); alpha_bad=-1 flips the conflicting component.
        primary_task: task key treated as primary (never modified).
        surgery_tasks: explicit list of aux tasks to modify; None = every
            task in ``task_losses`` except the primary.
        warmup_iters: standard (single-backward) training before surgery.
        buffer_dtype: 'float32' or 'float16' storage for the ring buffer.
        log_interval: emit decomposition statistics every N iters.
        grad_clip: usual mmcv grad-clip config.
    """

    def __init__(
        self,
        shared_layers: List[str],
        mode: str = 'svd_buffer',
        buffer_size: int = 16,
        svd_energy: float = 0.90,
        k_max: int = 8,
        alpha_good: float = 1.0,
        alpha_bad: float = -1.0,
        alpha_neutral: float = 1.0,
        primary_task: str = 'plan',
        surgery_tasks: Optional[List[str]] = None,
        warmup_iters: int = 500,
        buffer_dtype: str = 'float32',
        accum_steps: int = 1,
        log_interval: int = 50,
        log_verbose: bool = False,
        dump_per_sample: bool = False,
        track_direction: bool = False,
        ema_beta: float = 0.99,
        grad_clip: Optional[dict] = None,
    ):
        super().__init__(grad_clip=grad_clip)
        assert mode in ('svd_buffer', 'rank1'), mode
        assert 0.0 < svd_energy <= 1.0, svd_energy
        assert accum_steps >= 1, accum_steps
        self.shared_layers = shared_layers
        self.mode = mode
        self.buffer_size = buffer_size
        self.svd_energy = svd_energy
        self.k_max = k_max
        self.alpha_good = alpha_good
        self.alpha_bad = alpha_bad
        self.alpha_neutral = alpha_neutral
        self.primary_task = primary_task
        self.surgery_tasks = surgery_tasks
        self.warmup_iters = warmup_iters
        self.buffer_dtype = getattr(torch, buffer_dtype)
        self.accum_steps = accum_steps
        self.log_interval = log_interval
        self.log_verbose = log_verbose
        self.dump_per_sample = dump_per_sample
        self.track_direction = track_direction
        self.ema_beta = ema_beta

        self._shared_params: Optional[List[nn.Parameter]] = None
        self._flat_dim = 0
        self._buffer: Optional[torch.Tensor] = None  # (N, d)
        self._buf_fill = 0
        self._buf_ptr = 0
        self._initialized = False
        # Gradient-accumulation window state.
        self._win_count = 0
        self._accum_shared: Optional[torch.Tensor] = None
        # Diagnostics state (subspace drift / plan coherence / dump handle).
        self._prev_U: Optional[torch.Tensor] = None
        self._prev_gp: Optional[torch.Tensor] = None
        self._dump_fh = None
        # Per-task direction convergence tracking (EMA of grad vector + norm,
        # and previous grad for step-to-step self-coherence).
        self._ema_g: Dict[str, torch.Tensor] = {}
        self._ema_norm: Dict[str, float] = {}
        self._prev_g: Dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------- setup

    def _lazy_init(self, model):
        if self._initialized:
            return
        param_groups, _ids = get_shared_parameters_grouped(
            model, self.shared_layers)
        # Fixed flattening order: group insertion order, params in list order.
        self._shared_params = [p for params in param_groups.values()
                               for p in params]
        self._flat_dim = sum(p.numel() for p in self._shared_params)
        device = self._shared_params[0].device
        if self.mode == 'svd_buffer':
            self._buffer = torch.zeros(
                self.buffer_size, self._flat_dim,
                dtype=self.buffer_dtype, device=device)
        logger.info(
            f"[ATTITTUD] mode={self.mode} shared groups="
            f"{list(param_groups.keys())} flat_dim={self._flat_dim:,} "
            f"buffer={'%d x %d (%s)' % (self.buffer_size, self._flat_dim, self.buffer_dtype) if self.mode == 'svd_buffer' else 'n/a'} "
            f"alphas=(good {self.alpha_good}, bad {self.alpha_bad}, "
            f"neutral {self.alpha_neutral})")
        self._initialized = True

    # -------------------------------------------------------- flat helpers

    def _zero_shared_grads(self):
        for p in self._shared_params:
            if p.grad is not None:
                p.grad.zero_()

    def _flatten_shared(self) -> torch.Tensor:
        chunks = []
        for p in self._shared_params:
            if p.grad is not None:
                chunks.append(p.grad.detach().reshape(-1).float())
            else:
                chunks.append(torch.zeros(
                    p.numel(), device=p.device, dtype=torch.float32))
        return torch.cat(chunks)

    def _assign_shared(self, flat: torch.Tensor):
        offset = 0
        for p in self._shared_params:
            n = p.numel()
            chunk = flat[offset:offset + n].reshape(p.shape).to(p.dtype)
            if p.grad is None:
                p.grad = chunk.clone()
            else:
                p.grad.data.copy_(chunk)
            offset += n

    # ---------------------------------------------------------- subspace

    def _push_buffer(self, g_p: torch.Tensor):
        self._buffer[self._buf_ptr].copy_(g_p.to(self.buffer_dtype))
        self._buf_ptr = (self._buf_ptr + 1) % self.buffer_size
        self._buf_fill = min(self._buf_fill + 1, self.buffer_size)

    def _compute_basis(self, g_p: torch.Tensor):
        """Return (U, k, evals): U is (d, k) with orthonormal columns,
        evals the descending Gram eigenvalue spectrum (None for rank1)."""
        eps = 1e-12
        if self.mode == 'rank1':
            norm = g_p.norm()
            if norm < eps:
                return None, 0, None
            return (g_p / norm).unsqueeze(1), 1, None

        n = self._buf_fill
        B = self._buffer[:n].float()  # (n, d)
        gram = B @ B.t()              # (n, n)
        evals, evecs = torch.linalg.eigh(gram)
        # eigh returns ascending; flip to descending.
        evals = evals.flip(0).clamp_min(0.0)
        evecs = evecs.flip(1)
        total = evals.sum()
        if total < eps:
            return None, 0, evals
        energy = torch.cumsum(evals, 0) / total
        k = int((energy < self.svd_energy).sum().item()) + 1
        k = min(k, self.k_max, int((evals > eps * total).sum().item()))
        if k <= 0:
            return None, 0, evals
        # Back-project Gram eigenvectors to parameter space and normalize.
        U = B.t() @ evecs[:, :k]                       # (d, k)
        U = U / evals[:k].sqrt().clamp_min(eps)        # orthonormal columns
        return U, k, evals

    # ------------------------------------------------------ diagnostics

    def _subspace_overlap(self, U_prev, U_cur) -> float:
        """Grassmann-style overlap in [0,1]: mean captured energy of one
        subspace's basis in the other. 1.0 = identical span."""
        if U_prev is None or U_cur is None:
            return float('nan')
        M = U_prev.t() @ U_cur          # (k_prev, k_cur)
        r = min(U_prev.shape[1], U_cur.shape[1])
        return (M.pow(2).sum() / max(r, 1)).item()

    def _direction_metrics(self, flat: Dict[str, torch.Tensor],
                           want_stats: bool) -> dict:
        """Update per-task gradient-direction EMAs and return convergence
        metrics. Every task gradient ``g_t`` updates a running mean vector
        ``m = beta*m + (1-beta)*g`` and mean-norm ``mn``. Reported per task:

          - ``coherence`` = ||m|| / mn  in [0,1]: 1 => every step points the
            same way (converged direction); ~0 => directions cancel (noise).
          - ``align_ema`` = cos(g_t, m): current step's alignment with the
            running-mean direction.
          - ``self_coherence`` = cos(g_t, g_{t-1}): step-to-step stability.

        Also returns pairwise cosine of the *mean* directions
        (``dirmean/{a}_{b}``): the structural, noise-averaged task relation,
        complementing the noisy per-step conflict matrix.
        """
        eps = 1e-12
        beta = self.ema_beta
        out: dict = {}
        for t, g in flat.items():
            gn = g.norm()
            if t not in self._ema_g:
                self._ema_g[t] = g.clone()
                self._ema_norm[t] = gn.item()
            else:
                self._ema_g[t].mul_(beta).add_(g, alpha=1.0 - beta)
                self._ema_norm[t] = beta * self._ema_norm[t] + (1.0 - beta) * gn.item()
            if want_stats:
                m = self._ema_g[t]
                mn = max(self._ema_norm[t], eps)
                out[f'attittud/coherence/{t}'] = (m.norm() / mn).item()
                out[f'attittud/align_ema/{t}'] = (
                    g.dot(m) / (gn * m.norm()).clamp_min(eps)).item()
                if t in self._prev_g:
                    pg = self._prev_g[t]
                    out[f'attittud/self_coherence/{t}'] = (
                        g.dot(pg) / (gn * pg.norm()).clamp_min(eps)).item()
            self._prev_g[t] = g.clone()
        # Pairwise cosine of mean directions (structural task relation).
        if want_stats:
            tasks = list(self._ema_g.keys())
            for ii in range(len(tasks)):
                for jj in range(ii + 1, len(tasks)):
                    a, b = tasks[ii], tasks[jj]
                    ma, mb = self._ema_g[a], self._ema_g[b]
                    out[f'attittud/dirmean/{a}_{b}'] = (
                        ma.dot(mb) / (ma.norm() * mb.norm()).clamp_min(eps)).item()
        return out

    def _write_record(self, runner, record: dict) -> None:
        """Append one per-sample JSON record to work_dir/attittud_persample.jsonl."""
        import json
        import os
        if self._dump_fh is None:
            path = os.path.join(getattr(runner, 'work_dir', '.'),
                                'attittud_persample.jsonl')
            self._dump_fh = open(path, 'a', buffering=1)
            logger.info(f"[ATTITTUD] per-sample dump -> {path}")
        self._dump_fh.write(json.dumps(record) + "\n")

    @staticmethod
    def _cur_token(runner):
        """Best-effort id of the current micro-sample. HiP-AD nuScenes
        img_metas carries no scene token, so fall back to the frame
        timestamp (unique per sample) / instance_id."""
        try:
            batch = getattr(runner, 'data_batch', None)
            metas = batch['img_metas']
            metas = metas.data if hasattr(metas, 'data') else metas
            while isinstance(metas, (list, tuple)):
                metas = metas[0]
            if isinstance(metas, dict):
                for key in ('token', 'sample_idx', 'timestamp'):
                    if metas.get(key) is not None:
                        return metas[key]
        except Exception:
            pass
        return None

    # ----------------------------------------------------------- ddp bits

    def _is_distributed(self, model) -> bool:
        return hasattr(model, 'no_sync')

    def _allreduce_grads(self, model):
        world_size = dist.get_world_size()
        for p in model.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
                p.grad.data /= world_size

    def before_train_iter(self, runner):
        model = runner.model
        if not self._is_distributed(model):
            return
        # Disable DDP autograd hooks on surgery iterations so multiple
        # backward passes don't trip "marked ready twice"; gradients are
        # manually all-reduced after surgery.
        model.require_backward_grad_sync = not self._should_surgery(runner)

    # ------------------------------------------------------------- main

    def _should_surgery(self, runner) -> bool:
        return runner.iter >= self.warmup_iters

    def after_train_iter(self, runner):
        """Windowed gradient accumulation over ``accum_steps`` micro-samples.

        Each micro-sample runs its own forward/backward and (post-warmup)
        ATTITTUD surgery, contributing a per-sample shared-gradient vector.
        Private parameters accumulate their per-sample grads in ``.grad``
        naturally across the window; the surgical shared vector is summed in
        ``self._accum_shared``. The optimizer steps once per window on the
        mean gradient, restoring the batch-averaging buffer that batch-1
        removes while preserving per-sample surgery.
        """
        model = runner.model
        self._lazy_init(model)

        # ---- window start: clear grads + accumulator ----
        if self._win_count == 0:
            runner.optimizer.zero_grad()
            self._accum_shared = None

        surgery = (self._should_surgery(runner)
                   and self.primary_task in runner.outputs.get('task_losses', {})
                   and len(runner.outputs.get('task_losses', {})) > 1)

        if surgery:
            ok, contrib = self._surgery_micro(runner)
        else:
            ok, contrib = self._standard_micro(runner)

        if not ok:
            # skip_bad_step already zeroed grads and reset banks: abort window.
            self._win_count = 0
            self._accum_shared = None
            return

        # Accumulate the per-sample shared contribution.
        self._accum_shared = (contrib if self._accum_shared is None
                              else self._accum_shared + contrib)
        self._win_count += 1

        if self._win_count < self.accum_steps:
            return  # keep accumulating

        # ---- window complete: assign mean gradient, clip, step ----
        n = self._win_count
        self._assign_shared(self._accum_shared)
        for p in model.parameters():
            if p.grad is not None:
                p.grad.div_(n)

        if self._is_distributed(model):
            self._allreduce_grads(model)

        if self.grad_clip is not None:
            grad_norm = self.clip_grads(model.parameters())
            if grad_norm is not None:
                if not torch.isfinite(grad_norm):
                    skip_bad_step(runner, "non-finite grad norm (window)")
                    self._win_count = 0
                    self._accum_shared = None
                    return
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                         runner.outputs['num_samples'])

        runner.optimizer.step()
        self._win_count = 0
        self._accum_shared = None

    def _standard_micro(self, runner):
        """Warmup / no-surgery micro: plain backward, isolate shared grad.

        Returns (ok, shared_contrib). Private grads stay accumulated in
        ``.grad``; the shared component is read out, zeroed from ``.grad``
        (re-applied from ``_accum_shared`` at step time), and returned.
        """
        loss = runner.outputs['loss']
        if not torch.isfinite(loss):
            skip_bad_step(runner, f"non-finite loss {loss.item()}")
            return False, None
        loss.backward()
        shared = self._flatten_shared()
        if not torch.isfinite(shared).all():
            skip_bad_step(runner, "non-finite shared gradient (standard)")
            return False, None
        self._zero_shared_grads()
        return True, shared

    def _surgery_micro(self, runner):
        """One ATTITTUD surgery micro-sample.

        Runs T per-task backwards, decomposes each aux gradient against the
        planning subspace, and returns (ok, merged_shared). Diagnostic stats
        are logged directly here (decoupled from the window step so they
        always land on ``log_interval`` iters); private params accumulate
        per-task grads in ``.grad``; only the shared contribution is returned
        for windowed accumulation.
        """
        model = runner.model
        task_losses = runner.outputs['task_losses']
        aux_loss = runner.outputs.get('aux_loss', None)
        has_aux = aux_loss is not None

        if self.surgery_tasks is not None:
            aux_tasks = [t for t in task_losses
                         if t != self.primary_task and t in self.surgery_tasks]
            passthrough = [t for t in task_losses
                           if t != self.primary_task and t not in aux_tasks]
        else:
            aux_tasks = [t for t in task_losses if t != self.primary_task]
            passthrough = []

        bad_loss = (not all(torch.isfinite(v).all()
                            for v in task_losses.values())
                    or (has_aux and not torch.isfinite(aux_loss).all()))
        if bad_loss:
            skip_bad_step(runner, "non-finite task loss")
            return False, None

        # ---- per-task backward, capture shared flat grads ----
        order = aux_tasks + passthrough + [self.primary_task]
        flat: Dict[str, torch.Tensor] = {}
        for i, task in enumerate(order):
            self._zero_shared_grads()
            retain = (i < len(order) - 1) or has_aux
            task_losses[task].backward(retain_graph=retain)
            flat[task] = self._flatten_shared()

        aux_flat = None
        if has_aux:
            self._zero_shared_grads()
            aux_loss.backward()
            aux_flat = self._flatten_shared()

        # Leave private params holding their accumulated per-task grads;
        # clear the shared slots (re-applied from the accumulator at step).
        self._zero_shared_grads()

        g_p = flat[self.primary_task]
        finite = all(torch.isfinite(v).all() for v in flat.values())
        if not finite or (aux_flat is not None
                          and not torch.isfinite(aux_flat).all()):
            skip_bad_step(runner, "non-finite shared gradient")
            return False, None

        if self.mode == 'svd_buffer':
            self._push_buffer(g_p)
        U, k, evals = self._compute_basis(g_p)

        eps = 1e-12
        should_log = (runner.iter % self.log_interval == 0)
        want_stats = should_log or self.dump_per_sample
        gp_norm = g_p.norm().item()
        stats = ({'attittud/k': float(k),
                  'attittud/plan_grad_norm': gp_norm} if should_log else None)
        if should_log and self.mode == 'svd_buffer':
            stats['attittud/buffer_fill'] = float(self._buf_fill)

        # Per-sample record (lightweight; written every micro when enabled).
        record = ({'iter': int(runner.iter), 'k': int(k),
                   'plan_grad_norm': gp_norm,
                   'buffer_fill': int(self._buf_fill)}
                  if self.dump_per_sample else None)
        if record is not None:
            tok = self._cur_token(runner)
            if tok is not None:
                record['token'] = tok
            lv = runner.outputs.get('log_vars', {}) or {}
            if 'ego_loss_status' in lv:
                record['ego_loss_status'] = float(lv['ego_loss_status'])
            if 'loss' in lv:
                record['loss'] = float(lv['loss'])

        merged = g_p.clone()
        c_p = U.t() @ g_p if U is not None else None
        for t in aux_tasks:
            g = flat[t]
            if U is not None:
                c = U.t() @ g
                agree = c * c_p
                good_c = torch.where(agree > 0, c, torch.zeros_like(c))
                bad_c = torch.where(agree < 0, c, torch.zeros_like(c))
                par_good = U @ good_c
                par_bad = U @ bad_c
                perp = g - U @ c
                g_mod = (self.alpha_good * par_good
                         + self.alpha_bad * par_bad
                         + self.alpha_neutral * perp)
                merged += g_mod
                if want_stats:
                    g_sq = g.dot(g).clamp_min(eps)
                    fg = (par_good.dot(par_good) / g_sq).item()
                    fb = (par_bad.dot(par_bad) / g_sq).item()
                    fn = (perp.dot(perp) / g_sq).item()
                    cosp = (g.dot(g_p)
                            / (g.norm() * g_p.norm()).clamp_min(eps)).item()
                    modr = ((g_mod - g).norm()
                            / g.norm().clamp_min(eps)).item()
                    if stats is not None:
                        stats[f'attittud/{t}/frac_good'] = fg
                        stats[f'attittud/{t}/frac_bad'] = fb
                        stats[f'attittud/{t}/frac_neutral'] = fn
                        stats[f'attittud/{t}/cos_plan'] = cosp
                        stats[f'attittud/{t}/mod_ratio'] = modr
                    if record is not None:
                        record[f'{t}_grad_norm'] = g.norm().item()
                        record[f'{t}_cos_plan'] = cosp
                        record[f'{t}_frac_bad'] = fb
                        record[f'{t}_mod_ratio'] = modr
            else:
                merged += g  # no basis: plain sum
                if record is not None:
                    record[f'{t}_grad_norm'] = g.norm().item()

        for t in passthrough:
            merged += flat[t]
        if aux_flat is not None:
            merged += aux_flat

        # ---- per-task direction convergence (EMA updated every micro) ----
        if self.track_direction:
            dir_stats = self._direction_metrics(flat, want_stats)
            if stats is not None:
                stats.update(dir_stats)
            if record is not None:
                for t in flat:
                    for key in ('coherence', 'align_ema', 'self_coherence'):
                        k_full = f'attittud/{key}/{t}'
                        if k_full in dir_stats:
                            record[f'{t}_{key}'] = dir_stats[k_full]

        # ---- verbose diagnostics (log windows only): conflict matrix,
        #      subspace spectrum/drift, plan coherence, surgery intensity ----
        if should_log and self.log_verbose and stats is not None:
            all_tasks = list(flat.keys())
            n_conf = n_pair = 0
            for ii in range(len(all_tasks)):
                for jj in range(ii + 1, len(all_tasks)):
                    a, b = all_tasks[ii], all_tasks[jj]
                    ga, gb = flat[a], flat[b]
                    denom = (ga.norm() * gb.norm()).clamp_min(eps)
                    cos = (ga.dot(gb) / denom).item()
                    stats[f'attittud/conflict/{a}_{b}/cos'] = cos
                    n_pair += 1
                    if cos < 0:
                        n_conf += 1
            stats['attittud/conflict/rate'] = n_conf / max(n_pair, 1)
            for t in all_tasks:
                stats[f'attittud/grad_norm/{t}'] = flat[t].norm().item()
            # subspace spectrum + drift + plan coherence
            if evals is not None and evals.numel() > 0:
                tot = evals.sum().clamp_min(eps)
                stats['attittud/spec/eig0_frac'] = (evals[0] / tot).item()
                stats['attittud/spec/eff_rank'] = (
                    (evals.sum() ** 2) / evals.pow(2).sum().clamp_min(eps)
                ).item()
            stats['attittud/subspace_drift'] = 1.0 - self._subspace_overlap(
                self._prev_U, U)
            if self._prev_gp is not None:
                stats['attittud/plan_coherence'] = (
                    g_p.dot(self._prev_gp)
                    / (g_p.norm() * self._prev_gp.norm()).clamp_min(eps)).item()
            raw_sum = g_p.clone()
            for t in aux_tasks + passthrough:
                raw_sum = raw_sum + flat[t]
            if aux_flat is not None:
                raw_sum = raw_sum + aux_flat
            dch = (merged.dot(raw_sum)
                   / (merged.norm() * raw_sum.norm()).clamp_min(eps)).item()
            stats['attittud/surgery/direction_change'] = 1.0 - dch

        # Update drift/coherence references (detached, kept off the graph).
        if U is not None:
            self._prev_U = U.detach()
        self._prev_gp = g_p.detach()

        if record is not None:
            self._write_record(runner, record)

        # Log diagnostics directly (decoupled from window step so they always
        # appear on log_interval iters regardless of accumulation alignment).
        if should_log and stats is not None:
            runner.log_buffer.update(stats, runner.outputs['num_samples'])

        return True, merged
