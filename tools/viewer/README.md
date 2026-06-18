# HiP-AD Scene Inspector (VAD-viewer-style)

Browser app: **left = 6 individual camera images**, **right = a client-side HTML5
Canvas BEV** (dark `#111`, mouse **wheel-zoom / drag-pan**, instant layer toggles) —
comparing `full` vs an ablated model (`no_det`/`no_map`/`no_motion`) on the same scene.

The BEV is drawn in the browser from a geometry JSON (`/api/geom`), so toggles/zoom/pan
are instant. Layers: LiDAR points, GT boxes, **predicted detection boxes** (full=cyan,
ablated=pink), **predicted map vectors**, **predicted motion** (top mode), GT/full/ablated
ego trajectories with collision markers (✕). View **side-by-side** (full | ablated) or
**overlay**.

## Run (host, env `hipad`)
```bash
conda activate hipad
python tools/viewer/server.py 8077      # open http://localhost:8077 (VS Code forwards the port)
```
Geometry per scene ~3s; first time a **map/motion** layer is enabled at an epoch it loads
that `results.pkl` into RAM (~25s once, then instant).

## Top navigation (filter scenes by the analysis strata)
**epoch** (1/3/9) · **ablate** (det/map/motion) · **density** (dense `near≥6` / sparse) ·
**maneuver** (straight / turn) · **map** (complex / simple) · **rank** (showcase / top C_col /
top C_L2 / top |C_L2|) · Prev/Next (← →). 2nd bar: **view** (side/overlay) + **layer checkboxes**.

Predicted det/map/motion come straight from each variant's `results.pkl` — **no retraining**.
(Only the 1-step probe mode needs a GPU re-run.)

Endpoints: `/` (page), `/api/scenes` (filtered+ranked list), `/img?token=&cam=` (camera jpg),
`/api/geom` (lidar+boxes+trajs, fast), `/api/percep` (pred map+motion, lazy `results.pkl`).

### Ranking filters
- **ablated collides, full safe (dense)** — the flagship view: scenes where removing
  the task makes the planner collide while `full` stays safe, densest first.
  (`det` @1ep here = the "detection prevents collisions in dense scenes" story.)
- **top C_col** — scenes the task helps most on safety.
- **top C_L2** — scenes the task helps most on accuracy.
- **top |C_L2|** — biggest full-vs-ablated divergence.

## Pieces (all gate-verified)
| file | role | gate |
|---|---|---|
| `lib_io.py` | sorted val infos + token map + slim prediction caches (memoized) | len==6019, token-sort |
| `metrics.py` | per-scene L2, exact `planning_eval` replica | **L2 @9ep full = 0.6457** ✓ |
| `collision.py` | per-scene `obj_box_col` reusing repo `PlanningMetric` + `fut_boxes` | **col @9ep full = 0.082%** ✓ |
| `render_scene.py` / `inspector.py` | BEV + before/after renderers | visual |
| `server.py` | stdlib `http.server` interactive UI (no deps) | end-to-end |

## Analyses (cache/*.csv)
- `task_epoch_summary.py` → `task_epoch_table.csv`: det/map/motion contribution to
  planning at 1/3/9ep, both L2 and collision, with scene-feature stratification.
- `collision_strata.py` → `collision_table.csv`: per-scene collision, all variants/epochs.
- `strata_analysis.py` → `strata_table_e9.csv`: per-scene L2 transfer vs scene features.

## Modes
- **TG (full vs ablated)** — working now, no GPU.
- **1-step probe (θ vs θ−α·g)** — deferred: needs `probe.py` re-run with a head
  forward-hook to dump predictions (GPU). Same inspector layout will host it.

## Frame conventions (verified)
LiDAR/box frame x=forward, y=left. BEV: forward→up, left→left. Planner `plan_temp_2hz`
is absolute; GT = `cumsum(gt_ego_fut_trajs)`. Collision frame: eval's double x-flip
cancels, so plan coords used as-is against `fut_boxes`.
