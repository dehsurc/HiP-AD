# E11 — Parasitic Planning via Aligned Auxiliary Gradients

**"Can the primary task (planning) be trained without ever applying its own
gradient — using only the planning-aligned part of the auxiliary gradients as
the update, with the planning gradient serving as a compass?"**

Branch `nusc/aligned-aux` (worktree `HiP-AD-aligned`, based on `revision`).
Start point: `ckpts/rev_nusc/70+stage2_3ep.pth` (the same checkpoint ATTITTUD
E10 finetuned from). Date: 2026-07-10.

---

## 1. Hypothesis (formalised)

Let θ_s be the shared representation, θ_p the planning head, θ_a the aux heads
(det / map / motion / ego). Define the shared-parameter gradients
`g_p = ∇_{θ_s} L_plan`, `g_a = ∇_{θ_s} L_aux`.

- **Freeze θ_p** (planning head never updated) **and freeze backbone + neck**.
- Compute `g_p` **as a reference direction only** (a compass); never apply it.
- Update θ_s using **only the planning-aligned part** of each `g_a`.
- Question: does `L_plan` still decrease? I.e. can planning be learned
  **parasitically**, purely through aligned auxiliary gradients?

First-order intuition: stepping `−η·g_a` with `⟨g_a, g_p⟩ > 0` gives
`⟨g_p, Δθ_s⟩ < 0`, so `L_plan` decreases *to first order*. The open question is
whether this survives beyond the first order without planning's own gradient —
or stalls at the first fixed point where every aligned aux direction is
exhausted (`g_p ⟂` all aligned aux).

## 2. Relation to prior work

| Work | Mechanism | Relation to E11 |
|---|---|---|
| **Du et al. 2018**, *Adapting Auxiliary Losses Using Gradient Similarity* (arXiv:1812.02224) | Gate an auxiliary loss by the sign of `cos(g_a, g_main)`; keeps updating the main task, with a convergence guarantee to a main-task critical point. | Exact prototype of our `gate_mode='cosine_gate'`. **Novelty: we never update the primary** (compass only) and freeze its head, so their convergence guarantee no longer applies — only first-order descent remains. |
| **Dery et al. 2021 ATTITTUD** (arXiv:2108.11346) | Decompose `g_a` against a subspace of the primary gradient into good / bad / neutral; scale each. | Our `gate_mode='component'` = ATTITTUD with `(α_good, α_bad, α_neutral) = (1, 0, 0)` **and the primary self-update removed**. |
| **PCGrad** (Yu et al. 2020, arXiv:2001.06782), **CAGrad** (Liu et al. 2021), **GradVaccine**, **Aligned-MTL** (Senushkin 2023), **FAMO** (Liu 2023) | Resolve inter-task gradient conflict; every task keeps its own gradient. | We do the opposite of "keep everyone": we *drop* the primary's gradient and keep only aligned aux. |
| **MAXL** (Liu 2019), **AuxiLearn** (Navon 2021) | Learn auxiliary tasks/weights to help a primary; primary is still directly optimised. | Same "aux helps primary" spirit, but they never freeze the primary. |
| **Ego-status shortcut** — AD-MLP (Zhai 2023), **BEV-Planner** (Li 2024) | Show nuScenes planning is largely predictable from ego status; perception aux may be weak planning signal. | Predicts a likely **negative**: if aux ⟂ planning, aligned-aux carries little planning signal. Our measured alignment (below) supports this. |

**Novelty verdict.** "Primary as compass only, never applied, head frozen" is,
to our knowledge, unexplored. Du-2018 is the closest but always updates the
main task. Outcome value either way: a positive result proves task coupling is
a usable *lever* (not merely diagnostic); a negative is the first clean
isolation of the aligned channel showing planning needs its own gradient.

## 3. Method

`AlignedAuxOptimizerHook` (`projects/mmdet3d_plugin/core/hooks/aligned_aux_optimizer_hook.py`),
inherits the ATTITTUD machinery (per-task `retain_graph` backward isolating the
shared-parameter gradient, plan-gradient ring-buffer subspace, batch-1
nan-guard + InstanceBank reset, gradient-accumulation window K=8, per-task
direction diagnostics).

**Clean causal isolation.** We freeze the planning head
(`plan_refine` / `plan_deformable` / `plan_instance_bank` / `plan_anchor_encoder`,
2.6 % of params) **plus the image backbone and neck**. The only trainable
*shared* representation left is the cross-task decoder ops
(`inter_gnn` / `ffn` / `norm`), which the surgery fully controls. Therefore:

- planning head frozen → planning never updated (head);
- backbone / neck frozen → the planning gradient cannot leak into shared
  features there;
- surgery excludes `g_p` on the decoder ops → planning never shapes the one
  remaining trainable shared set;
- aux private heads (det / map / motion / ego) keep training on their own loss.

So the shared representation is shaped **only** by planning-aligned auxiliary
gradients, never by planning itself. *(Limitation: any minor decoder op that is
shared, trainable, and outside `{inter_gnn, ffn, norm}` still receives the
summed multi-task gradient including planning's; this residual is second-order
and noted for follow-up.)*

**Gate modes.**
- `cosine_gate` (**primary**, Du-2018): keep the whole `g_a` iff
  `cos(g_a, g_p) > 0`, else drop it — magnitude preserving.
- `component` (ATTITTUD projection): keep only the in-subspace, sign-agreeing
  part; `renorm='aux_full'` rescales the assembled aligned direction to the
  natural aux magnitude to separate *direction* from *magnitude*.
- `full`: all aux untouched (upper-reference control).
- `gate_sign = −1`: keep the **conflicting** aux instead (anti-aligned control).

**Go / no-go logging** (per 50 iters): `aligned/energy_ratio`
(‖aligned Σ‖ / ‖raw aux Σ‖), `aligned/gate_rate` (fraction of aux kept),
`aligned/span_cover` (how well aux directions cover the plan subspace),
`aligned/*/cos_plan`, plus inherited coherence/subspace metrics.

## 4. Experimental design

All arms: batch-1, fp32, 3 epochs (≈84.4 k iters) from `70+stage2_3ep`, same
freeze regime, LR 7e-5 (√-scaled to eff-batch 8), single GPU.

| Arm | Config | Setting | Tests |
|---|---|---|---|
| (i) **aligned** | `E11_aligned_gated` | cosine_gate, sign +1 | can aligned aux train planning? |
| (ii) all-aux | `E11_ctrl_allaux` | gate_mode=full | does *any* shared-feature change help, regardless of alignment? |
| (iii) anti-aligned | `E11_ctrl_antialign` | cosine_gate, sign −1 | is the alignment **sign** causal? |
| (iv) floor | — (eval of `70+stage2_3ep`) | no shared learning for planning | starting planning L2 = **1.0337** |
| (v) strict projection | `E11_aligned_component` | ATTITTUD component + renorm | is the aligned *direction* useful when given real magnitude? |

**Decision tree (planning L2 vs floor 1.0337):**
- (i) ≈ (ii) and both < floor → **parasitic learning confirmed; coupling is a lever** (headline positive).
- (i) < floor but (i) worse than (ii) → **partial parasitism**; alignment selects a sub-optimal but useful subset.
- (i) ≈ floor (flat) → aligned channel inert; **planning needs its own gradient** (clean negative).
- (i) worse than (iii) → the alignment **sign is causal** (aligned helps, conflicting hurts).

Key contrasts: **(i) vs (iv)** = does the aligned channel have any force; **(i) vs (ii)** = aligned selection vs generic feature change.

## 5. Prior from the ATTITTUD run (go/no-go)

From E10 ATTITTUD training logs (same start checkpoint), the planning-aligned
fraction of each aux gradient is **tiny**:

- `frac_good` ≈ **0.03–0.10 %** for det / map / motion (aux ⟂ plan);
- `cos_plan` ≈ ±0.01 for det / map / motion (essentially orthogonal);
- **ego** is the exception: `frac_good` ≈ 7–10 %, `cos_plan` up to ≈ 0.10–0.12,
  and `dirmean(ego,plan)` ≈ 0.16–0.22 — ego carries almost all of the aligned
  signal (unsurprising: ego-status is planning-adjacent).

**Prediction.** Under strict projection the aligned energy is ~sub-percent →
near-inert. `cosine_gate` (magnitude-preserving) gives the hypothesis its best,
literature-grounded shot: on any step it keeps whichever aux are aligned at full
magnitude (~half of pairs, given the near-zero mean cosine), so the shared
decoder ops *do* move — the test is then genuinely whether that aligned motion
improves planning or is just noise. Given aux ⟂ plan and the ego-status
shortcut literature, a **clean negative is the most likely outcome**, with any
positive effect expected to be ego-driven.

## 6. Status (2026-07-10, running)

- Worktree `HiP-AD-aligned` on `nusc/aligned-aux` (based on `revision` +
  cherry-picked ATTITTUD infra `209013d` + aligned-aux commit `c02cf7a`).
- Code import-verified; all 4 configs parse; hook registered.
- 60-iter smoke passed: exit 0, finite loss, hook active (aligned/* metrics
  logged). GPU compute was free during the ATTITTUD eval's CPU-only map-mAP
  phase, so training started in parallel rather than waiting.
- **Primary run `E11_aligned_gated` LAUNCHED** (tmux `e11_gated`, GPU0).
- **Freeze verified at runtime** (from the training log, `before_run`):
  froze **428 tensors / 36.04 M params** (backbone + neck + planning head);
  **trainable 55.29 M / 91.69 M = 60.3 %**; `planning head fully frozen
  (0 leaks)`. → planning is genuinely never updated.
- Controls (ii)/(iii)/(v) queued (single GPU; run sequentially).

### 6.1 Early training observations (iter ≤ 350, healthy)

| metric | value | reading |
|---|---|---|
| `nan_guard/skipped` | **0** | batch-1 stable (planning+backbone frozen removes the ego-divergence pathway) |
| `loss` | 30.3 → 27.0 (↓) | aux tasks are learning |
| `ego_loss_status` | ~13 (stable) | no ego blow-up (cf. ATTITTUD needed surgery to tame this) |
| `aligned/gate_rate` | ~0.50–0.57 | ~half of aux kept per step (near-zero mean cosine → coin-flip gate) |
| `aligned/energy_ratio` | ~0.72–0.80 | the kept (aligned) aux carry ~¾ of the raw aux magnitude → shared ops **do** move substantially |
| `aligned/k` | ~4–4.6 | plan subspace rank (buffer filled; matches ATTITTUD k≈4–6) |
| `aligned/plan_grad_norm` | ~10–12 | compass magnitude, non-trivial |
| **`aligned/span_cover`** | **~0.005–0.007** | **aux directions barely cover the plan subspace (<1%)** |

**Key early signal.** The shared decoder ops are being pushed hard
(`energy_ratio` ≈ 0.8), but almost orthogonally to what planning needs
(`span_cover` ≈ 0.6 %). This is the aux ⟂ plan prior showing up live: the
aligned-gate keeps large-magnitude aux updates whose overlap with the planning
descent subspace is negligible. It foreshadows the **likely negative** (planning
changes will be incidental, not directed), while confirming the run is healthy
and the aux tasks themselves are training normally.

## 7. Results

*(to be filled after training + post-hoc eval — `L2`, `mAP_normal`, collision,
and the go/no-go trajectories `aligned/energy_ratio`, `aligned/gate_rate`,
`aligned/ego/cos_plan`.)*

| Arm | plan L2 (avg) | mAP_normal | obj_col | vs floor 1.0337 |
|---|---|---|---|---|
| (iv) floor (3ep ckpt) | 1.0337 | 0.4988 | 0.0089 | — |
| (i) aligned_gated | _pending_ | | | |
| (ii) all-aux | _pending_ | | | |
| (iii) anti-aligned | _pending_ | | | |
| (v) component | _pending_ | | | |
