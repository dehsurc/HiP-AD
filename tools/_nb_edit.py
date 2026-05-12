"""Notebook editor for the plan-centric refactor.

Usage: python tools/_nb_edit.py <command>

Commands are defined per task and mutate the notebook JSON in place.
"""
import argparse
import json
from pathlib import Path

NB_PATH = Path("docs/superpowers/specs/2026-05-02-gradient-dynamics-diagnostic-framework/phase2_hipad_vad_analysis_visualization.ipynb")


def load_nb():
    return json.loads(NB_PATH.read_text())


def save_nb(nb):
    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")


def code_cell(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(keepends=True)}


def md_cell(src):
    return {"cell_type": "markdown", "metadata": {},
            "source": src.splitlines(keepends=True)}


def _insert_after(nb, index, cells):
    nb["cells"][index + 1:index + 1] = cells


def cmd_task16():
    nb = load_nb()
    src = "".join(nb["cells"][3]["source"])
    addition = "\n\nfrom tools.gradient_analysis.plan_centric import (\n    ensure_columns,\n    bootstrap_ci,\n    practical_threshold,\n    standardize_probe_df,\n    get_probe_base,\n)\n\nprobe = standardize_probe_df(probe)\nbase = get_probe_base(probe, variant='normalized', steps=1)\nprint('base rows:', len(base))\n"
    if "standardize_probe_df(probe)" not in src:
        nb["cells"][3]["source"] = (src + addition).splitlines(keepends=True)
    save_nb(nb)
    print("task16 applied")


CMDS = {"task16": cmd_task16}


def cmd_task17():
    nb = load_nb()
    md = md_cell("# Part F. Plan-centric directed transfer\n\n"
                 "Question: 각 task의 1-step update가 planning loss를 줄이는가?\n"
                 "Source: `det`, `map`, `motion`, `plan`. Target: `plan`.\n")
    code_compute = code_cell(
        "from pathlib import Path\n"
        "from tools.gradient_analysis.plan_centric import (\n"
        "    build_plan_transfer_summary, top_beneficial, top_harmful,\n"
        ")\n"
        "OUT_DIR = Path('gradient_analysis_results/plan_centric')\n"
        "OUT_DIR.mkdir(parents=True, exist_ok=True)\n"
        "(OUT_DIR / 'figures/F').mkdir(parents=True, exist_ok=True)\n"
        "plan_transfer = build_plan_transfer_summary(base)\n"
        "plan_transfer.to_csv(OUT_DIR / 'plan_transfer_summary.csv', index=False)\n"
        "display(plan_transfer.head())\n"
    )
    code_heatmap = code_cell(
        "import matplotlib.pyplot as plt\n"
        "SRC_ORDER = ['det', 'map', 'motion', 'plan']\n"
        "for (model, ckpt), sub in plan_transfer.groupby(['model', 'checkpoint']):\n"
        "    for metric in ['mean_gain', 'practical_helpful_rate']:\n"
        "        pivot = sub.pivot_table(index='layer', columns='source_task', values=metric).reindex(columns=SRC_ORDER)\n"
        "        fig, ax = plt.subplots(figsize=(6, max(3, 0.3 * len(pivot))))\n"
        "        cmap = 'RdBu_r' if metric == 'mean_gain' else 'viridis'\n"
        "        im = ax.imshow(pivot.values, aspect='auto', cmap=cmap)\n"
        "        ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns)\n"
        "        ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index, fontsize=8)\n"
        "        ax.set_title(f'{model} {ckpt}: target=plan, {metric}')\n"
        "        fig.colorbar(im, ax=ax)\n"
        "        fig.tight_layout()\n"
        "        fig.savefig(OUT_DIR / f'figures/F/{model}_{ckpt}_{metric}.png', dpi=120)\n"
        "        plt.show()\n"
    )
    code_tables = code_cell(
        "top_ben = top_beneficial(plan_transfer, k=10, by='mean_gain')\n"
        "top_ben.to_csv(OUT_DIR / 'plan_transfer_top_beneficial.csv', index=False)\n"
        "display(top_ben)\n"
        "top_harm = top_harmful(plan_transfer, k=10, by='mean_gain')\n"
        "top_harm.to_csv(OUT_DIR / 'plan_transfer_top_harmful.csv', index=False)\n"
        "display(top_harm)\n"
    )
    _insert_after(nb, 3, [md, code_compute, code_heatmap, code_tables])
    save_nb(nb)
    print("task17 applied")


CMDS["task17"] = cmd_task17


def cmd_task18():
    nb = load_nb()
    end = next(i for i, c in enumerate(nb["cells"])
               if c["cell_type"] == "code"
               and "plan_transfer_top_harmful" in "".join(c["source"]))
    md = md_cell("# Part G. Directed asymmetry around planning\n\n"
                 "`A → plan` vs `plan → A`, A ∈ {det, map, motion}.\n")
    code_compute = code_cell(
        "from tools.gradient_analysis.plan_centric import build_asymmetry_summary\n"
        "(OUT_DIR / 'figures/G').mkdir(parents=True, exist_ok=True)\n"
        "asym = build_asymmetry_summary(base)\n"
        "asym.to_csv(OUT_DIR / 'plan_asymmetry_summary.csv', index=False)\n"
        "asym[['model','checkpoint','layer','aux_task','asymmetry','plan_transfer_ratio']].to_csv(\n"
        "    OUT_DIR / 'plan_transfer_ratio_summary.csv', index=False)\n"
        "display(asym.head())\n"
    )
    code_heatmap = code_cell(
        "AUX_ORDER = ['det', 'map', 'motion']\n"
        "for (model, ckpt), sub in asym.groupby(['model', 'checkpoint']):\n"
        "    for metric in ['asymmetry', 'plan_transfer_ratio']:\n"
        "        pivot = sub.pivot_table(index='layer', columns='aux_task', values=metric).reindex(columns=AUX_ORDER)\n"
        "        fig, ax = plt.subplots(figsize=(5, max(3, 0.3 * len(pivot))))\n"
        "        im = ax.imshow(pivot.values, aspect='auto', cmap='RdBu_r')\n"
        "        ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns)\n"
        "        ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index, fontsize=8)\n"
        "        ax.set_title(f'{model} {ckpt}: {metric}')\n"
        "        fig.colorbar(im, ax=ax); fig.tight_layout()\n"
        "        fig.savefig(OUT_DIR / f'figures/G/{model}_{ckpt}_{metric}.png', dpi=120)\n"
        "        plt.show()\n"
    )
    code_trend = code_cell(
        "for model, sub in asym.groupby('model'):\n"
        "    for metric in ['asymmetry', 'plan_transfer_ratio']:\n"
        "        fig, ax = plt.subplots(figsize=(7, 4))\n"
        "        for aux, sub2 in sub.groupby('aux_task'):\n"
        "            agg = sub2.groupby('checkpoint_order')[metric].mean()\n"
        "            ax.plot(agg.index, agg.values, marker='o', label=aux)\n"
        "        ax.axhline(0, color='gray', lw=0.5)\n"
        "        ax.set_xlabel('checkpoint_order'); ax.set_ylabel(metric)\n"
        "        ax.set_title(f'{model}: {metric} trend')\n"
        "        ax.legend(); fig.tight_layout()\n"
        "        fig.savefig(OUT_DIR / f'figures/G/{model}_{metric}_trend.png', dpi=120)\n"
        "        plt.show()\n"
    )
    code_interp = code_cell(
        "display(asym[['model','checkpoint','layer','aux_task','interpretation']].head(20))\n"
    )
    _insert_after(nb, end, [md, code_compute, code_heatmap, code_trend, code_interp])
    save_nb(nb)
    print("task18 applied")


CMDS["task18"] = cmd_task18


def cmd_task19():
    nb = load_nb()
    end = next(i for i, c in enumerate(nb["cells"])
               if c["cell_type"] == "code"
               and "asym[['model','checkpoint','layer','aux_task','interpretation']]" in "".join(c["source"]))
    md = md_cell("# Part H. Effect-size-aware 1-step probe\n\n"
                 "Helpful ratio alone overweights noisy near-zero deltas. "
                 "Effect size = mean_gain / std_delta.\n")
    code_compute = code_cell(
        "from tools.gradient_analysis.plan_centric import build_effect_size_summary\n"
        "(OUT_DIR / 'figures/H').mkdir(parents=True, exist_ok=True)\n"
        "effect = build_effect_size_summary(base, target='plan')\n"
        "effect.to_csv(OUT_DIR / 'effect_size_probe_summary.csv', index=False)\n"
        "effect[['model','checkpoint','layer','source_task','helpful_rate','mean_gain','effect_size']].to_csv(\n"
        "    OUT_DIR / 'helpful_vs_effect_size.csv', index=False)\n"
        "display(effect.head())\n"
    )
    code_scatter = code_cell(
        "fig, ax = plt.subplots(figsize=(7, 5))\n"
        "colors = {'det':'C0','map':'C1','motion':'C2','plan':'C3'}\n"
        "markers = {'HiP-AD':'o','VAD':'s'}\n"
        "for (model, src), sub in effect.groupby(['model','source_task']):\n"
        "    ax.scatter(sub['helpful_rate'], sub['mean_gain'],\n"
        "               c=colors.get(src,'gray'), marker=markers.get(model,'x'),\n"
        "               alpha=0.6, label=f'{model}/{src}')\n"
        "top = effect.reindex(effect['mean_gain'].abs().sort_values(ascending=False).index).head(15)\n"
        "for _, r in top.iterrows():\n"
        "    ax.annotate(f\"{r['layer']}/{r['source_task'][:3]}\", (r['helpful_rate'], r['mean_gain']),\n"
        "                fontsize=7, alpha=0.7)\n"
        "ax.axhline(0, color='gray', lw=0.5); ax.axvline(0.5, color='gray', lw=0.5)\n"
        "ax.set_xlabel('helpful_rate'); ax.set_ylabel('mean_gain (plan)')\n"
        "ax.set_title('helpful_rate vs mean_gain (target=plan)')\n"
        "ax.legend(fontsize=7, ncol=2); fig.tight_layout()\n"
        "fig.savefig(OUT_DIR / 'figures/H/helpful_vs_gain_scatter.png', dpi=120)\n"
        "plt.show()\n"
    )
    code_ci = code_cell(
        "import numpy as np\n"
        "for model, sub in effect.groupby('model'):\n"
        "    fig, ax = plt.subplots(figsize=(8, 4))\n"
        "    pos = 0; xticks = []; xlabels = []\n"
        "    for src, sub2 in sub.groupby('source_task'):\n"
        "        top = sub2.reindex(sub2['mean_gain'].abs().sort_values(ascending=False).index).head(5)\n"
        "        for _, r in top.iterrows():\n"
        "            ax.errorbar(pos, r['mean_gain'],\n"
        "                        yerr=[[r['mean_gain']-r['ci_lo_gain']],[r['ci_hi_gain']-r['mean_gain']]],\n"
        "                        fmt='o', color=colors.get(src,'gray'))\n"
        "            xticks.append(pos); xlabels.append(f\"{src[:3]}/{r['layer'][:10]}\"); pos += 1\n"
        "        pos += 1\n"
        "    ax.axhline(0, color='gray', lw=0.5)\n"
        "    ax.set_xticks(xticks); ax.set_xticklabels(xlabels, rotation=60, ha='right', fontsize=7)\n"
        "    ax.set_ylabel('mean_gain ± 95% CI'); ax.set_title(f'{model}: top influential layers')\n"
        "    fig.tight_layout(); fig.savefig(OUT_DIR / f'figures/H/{model}_top_layers_ci.png', dpi=120)\n"
        "    plt.show()\n"
    )
    code_flags = code_cell(
        "flag_cols = ['flag_high_helpful_low_gain','flag_positive_but_insig',\n"
        "             'flag_high_practical_helpful','flag_high_large_harm']\n"
        "for fc in flag_cols:\n"
        "    flagged = effect[effect[fc]]\n"
        "    print(f'{fc}: {len(flagged)} rows')\n"
        "    if not flagged.empty:\n"
        "        display(flagged[['model','checkpoint','layer','source_task','mean_gain','helpful_rate']].head(10))\n"
    )
    _insert_after(nb, end, [md, code_compute, code_scatter, code_ci, code_flags])
    save_nb(nb)
    print("task19 applied")


CMDS["task19"] = cmd_task19


def cmd_task20():
    nb = load_nb()
    end = next(i for i, c in enumerate(nb["cells"])
               if c["cell_type"] == "code"
               and "flag_high_large_harm" in "".join(c["source"]))
    md = md_cell("# Part I. First-order prediction vs actual 1-step effect\n\n"
                 "Checks whether actual ΔL matches the linear prediction "
                 "`-step_size · <g_src, g_tgt>`.\n")
    code_main = code_cell(
        "from tools.gradient_analysis.plan_centric import (\n"
        "    detect_first_order_columns, build_first_order_residual_summary,\n"
        ")\n"
        "(OUT_DIR / 'figures/I').mkdir(parents=True, exist_ok=True)\n"
        "fo_cols = detect_first_order_columns(probe)\n"
        "if fo_cols is None:\n"
        "    print('SKIPPED. Required columns for first-order analysis:')\n"
        "    print('- source_task\\n- target_task\\n- model\\n- checkpoint\\n- layer\\n- delta')\n"
        "    print('- grad_dot (or equivalent)\\n- step_size / lr / probe_lr')\n"
        "    first_order = None\n"
        "else:\n"
        "    first_order = build_first_order_residual_summary(base, fo_cols)\n"
        "    first_order.to_csv(OUT_DIR / 'first_order_vs_actual_summary.csv', index=False)\n"
        "    first_order[first_order['target_task']=='plan'].to_csv(\n"
        "        OUT_DIR / 'first_order_residual_by_layer.csv', index=False)\n"
        "    display(first_order.head())\n"
    )
    code_plots = code_cell(
        "import numpy as np\n"
        "if first_order is not None:\n"
        "    df_full = base.copy()\n"
        "    df_full['pred_delta'] = -df_full[fo_cols['step_size']] * df_full[fo_cols['grad_dot']]\n"
        "    df_full['residual'] = df_full['delta'] - df_full['pred_delta']\n"
        "    fig, ax = plt.subplots(figsize=(5, 5))\n"
        "    ax.scatter(df_full['pred_delta'], df_full['delta'], alpha=0.4, s=8)\n"
        "    lim = max(abs(df_full['pred_delta'].min()), abs(df_full['delta'].max()))\n"
        "    ax.plot([-lim, lim], [-lim, lim], 'r--', lw=0.5)\n"
        "    ax.set_xlabel('predicted -lr·<g_s,g_t>'); ax.set_ylabel('actual ΔL')\n"
        "    ax.set_title('Actual vs first-order prediction')\n"
        "    fig.tight_layout(); fig.savefig(OUT_DIR / 'figures/I/actual_vs_pred.png', dpi=120)\n"
        "    plt.show()\n"
        "    plan_res = first_order[first_order['target_task'] == 'plan']\n"
        "    for (model, ckpt), sub in plan_res.groupby(['model', 'checkpoint']):\n"
        "        pivot = sub.pivot_table(index='layer', columns='source_task', values='mean_residual').reindex(columns=['det','map','motion','plan'])\n"
        "        fig, ax = plt.subplots(figsize=(5, max(3, 0.3*len(pivot))))\n"
        "        im = ax.imshow(pivot.values, aspect='auto', cmap='RdBu_r')\n"
        "        ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns)\n"
        "        ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index, fontsize=8)\n"
        "        ax.set_title(f'{model} {ckpt}: residual (plan target)')\n"
        "        fig.colorbar(im, ax=ax); fig.tight_layout()\n"
        "        fig.savefig(OUT_DIR / f'figures/I/{model}_{ckpt}_residual_plan.png', dpi=120)\n"
        "        plt.show()\n"
        "    for model, sub in df_full.groupby('model'):\n"
        "        fig, ax = plt.subplots(figsize=(6, 3))\n"
        "        ax.hist(sub['residual'], bins=60)\n"
        "        ax.set_title(f'{model}: residual distribution'); ax.set_xlabel('actual - predicted')\n"
        "        fig.tight_layout(); fig.savefig(OUT_DIR / f'figures/I/{model}_residual_hist.png', dpi=120)\n"
        "        plt.show()\n"
    )
    _insert_after(nb, end, [md, code_main, code_plots])
    save_nb(nb)
    print("task20 applied")


CMDS["task20"] = cmd_task20


def cmd_task21():
    nb = load_nb()
    end = next(i for i, c in enumerate(nb["cells"])
               if c["cell_type"] == "code"
               and "first_order is not None" in "".join(c["source"]))
    md = md_cell("# Part J. Planning sensitivity and query reachability\n\n"
                 "Loads `query_sensitivity.csv` if produced by the runner. "
                 "Missing file → template + SKIP.\n")
    code_main = code_cell(
        "import os\n"
        "from tools.gradient_analysis.plan_centric import (\n"
        "    load_or_template_query_sensitivity, build_query_sensitivity_summary,\n"
        ")\n"
        "(OUT_DIR / 'figures/J').mkdir(parents=True, exist_ok=True)\n"
        "QUERY_SENSITIVITY_PATH = Path(os.environ.get(\n"
        "    'QUERY_SENSITIVITY_PATH', OUT_DIR / 'query_sensitivity.csv'))\n"
        "qs_df, status = load_or_template_query_sensitivity(QUERY_SENSITIVITY_PATH)\n"
        "if status == 'template':\n"
        "    print(f'SKIPPED. Template written to {QUERY_SENSITIVITY_PATH}.')\n"
        "    print('Populate via tools/gradient_analysis/query_sensitivity.run_query_sensitivity')\n"
        "    qs_summary = None\n"
        "else:\n"
        "    qs_summary = build_query_sensitivity_summary(qs_df)\n"
        "    qs_summary.to_csv(OUT_DIR / 'query_sensitivity_summary.csv', index=False)\n"
        "    display(qs_summary.head())\n"
    )
    code_plots = code_cell(
        "if qs_summary is not None and not qs_summary.empty:\n"
        "    TASK_ORDER = ['det','map','motion','plan']\n"
        "    for (model, ckpt), sub in qs_summary.groupby(['model','checkpoint']):\n"
        "        agg = sub.groupby('task_query_type')['mean_sensitivity'].mean().reindex(TASK_ORDER)\n"
        "        fig, ax = plt.subplots(figsize=(5, 3))\n"
        "        ax.bar(agg.index, agg.values)\n"
        "        ax.set_title(f'{model} {ckpt}: mean planning sensitivity by query task')\n"
        "        fig.tight_layout(); fig.savefig(OUT_DIR / f'figures/J/{model}_{ckpt}_sens_bar.png', dpi=120)\n"
        "        plt.show()\n"
        "        pivot = sub.pivot_table(index='layer', columns='task_query_type',\n"
        "                                values='mean_sensitivity').reindex(columns=TASK_ORDER)\n"
        "        if not pivot.empty:\n"
        "            fig, ax = plt.subplots(figsize=(5, max(3, 0.3 * len(pivot))))\n"
        "            im = ax.imshow(pivot.values, aspect='auto', cmap='viridis')\n"
        "            ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns)\n"
        "            ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index, fontsize=8)\n"
        "            ax.set_title(f'{model} {ckpt}: layer × task_query_type sensitivity')\n"
        "            fig.colorbar(im, ax=ax); fig.tight_layout()\n"
        "            fig.savefig(OUT_DIR / f'figures/J/{model}_{ckpt}_sens_heatmap.png', dpi=120)\n"
        "            plt.show()\n"
        "    for model, sub in qs_summary.groupby('model'):\n"
        "        fig, ax = plt.subplots(figsize=(6, 3))\n"
        "        for tq, sub2 in sub.groupby('task_query_type'):\n"
        "            agg = sub2.groupby('checkpoint_order')['mean_sensitivity'].mean()\n"
        "            ax.plot(agg.index, agg.values, marker='o', label=tq)\n"
        "        ax.set_xlabel('checkpoint_order'); ax.set_ylabel('mean_sensitivity')\n"
        "        ax.set_title(f'{model}: planning sensitivity over training')\n"
        "        ax.legend(); fig.tight_layout()\n"
        "        fig.savefig(OUT_DIR / f'figures/J/{model}_sens_trend.png', dpi=120)\n"
        "        plt.show()\n"
    )
    _insert_after(nb, end, [md, code_main, code_plots])
    save_nb(nb)
    print("task21 applied")


CMDS["task21"] = cmd_task21


def cmd_task22():
    nb = load_nb()
    end = next(i for i, c in enumerate(nb["cells"])
               if c["cell_type"] == "code"
               and "qs_summary is not None and not qs_summary.empty" in "".join(c["source"]))
    md = md_cell("# Part K. Task loss weight elasticity\n\n"
                 "Loads `task_weight_elasticity_runs.csv`. Missing file → "
                 "template + SKIP.\n")
    code_main = code_cell(
        "import os\n"
        "from tools.gradient_analysis.plan_centric import (\n"
        "    load_or_template_elasticity, build_elasticity_summary,\n"
        "    planning_safe_weight_range,\n"
        ")\n"
        "(OUT_DIR / 'figures/K').mkdir(parents=True, exist_ok=True)\n"
        "ELASTICITY_LOG_PATH = Path(os.environ.get(\n"
        "    'ELASTICITY_LOG_PATH', OUT_DIR / 'task_weight_elasticity_runs.csv'))\n"
        "el_df, status = load_or_template_elasticity(ELASTICITY_LOG_PATH)\n"
        "if status == 'template':\n"
        "    print(f'SKIPPED. Template written to {ELASTICITY_LOG_PATH}.')\n"
        "    elasticity = None\n"
        "else:\n"
        "    elasticity = build_elasticity_summary(el_df)\n"
        "    elasticity.to_csv(OUT_DIR / 'task_weight_elasticity_summary.csv', index=False)\n"
        "    safe = planning_safe_weight_range(elasticity, tol=0.01)\n"
        "    safe.to_csv(OUT_DIR / 'planning_safe_weight_range.csv', index=False)\n"
        "    display(elasticity.head())\n"
    )
    code_plots = code_cell(
        "if elasticity is not None and not elasticity.empty:\n"
        "    for swept, sub in elasticity.groupby('swept_task'):\n"
        "        fig, axes = plt.subplots(1, 2, figsize=(10, 3))\n"
        "        for model, sub2 in sub.groupby('model'):\n"
        "            sub2 = sub2.sort_values('log_lambda')\n"
        "            axes[0].plot(sub2['log_lambda'], sub2['task_metric'], 'o-', label=model)\n"
        "            axes[1].plot(sub2['log_lambda'], sub2['plan_metric'], 'o-', label=model)\n"
        "        axes[0].set_title(f'lambda_{swept} vs task_metric'); axes[0].legend()\n"
        "        axes[1].set_title(f'lambda_{swept} vs plan_metric'); axes[1].legend()\n"
        "        axes[0].set_xlabel('log(lambda)'); axes[1].set_xlabel('log(lambda)')\n"
        "        fig.tight_layout(); fig.savefig(OUT_DIR / f'figures/K/{swept}_lambda_curves.png', dpi=120)\n"
        "        plt.show()\n"
        "    fig, ax = plt.subplots(figsize=(6, 4))\n"
        "    for (swept, model), sub in elasticity.groupby(['swept_task','model']):\n"
        "        ax.scatter(sub['relative_task_metric'], sub['relative_plan_metric'],\n"
        "                   label=f'{model}/{swept}', s=30)\n"
        "    ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)\n"
        "    ax.set_xlabel('relative task gain (+ better)')\n"
        "    ax.set_ylabel('relative plan degradation (+ worse for plan_l2)')\n"
        "    ax.set_title('Pareto: task gain vs plan degradation')\n"
        "    ax.legend(fontsize=7); fig.tight_layout()\n"
        "    fig.savefig(OUT_DIR / 'figures/K/pareto.png', dpi=120); plt.show()\n"
        "    display(safe[safe['planning_safe']])\n"
    )
    _insert_after(nb, end, [md, code_main, code_plots])
    save_nb(nb)
    print("task22 applied")


CMDS["task22"] = cmd_task22


_DEMOTE_HEADERS = [
    ("# Part A. HiP-AD 세부 분석",                    "# Appendix.A. HiP-AD diagnostic (legacy)"),
    ("# Part B. VAD 세부 분석",                       "# Appendix.B. VAD diagnostic (legacy)"),
    ("# Part C. 1-step probe를 task 중심으로 다시 비교", "# Appendix.C. Task-pair helpful matrix (Appendix only)"),
    ("# Part D. Distribution과 correlation 보조 분석",  "# Appendix.D. Distribution / correlation (legacy)"),
    ("# Part E. 기존 `gradient_analysis_viz.ipynb` 스타일 추가 분석",
     "# Appendix.E. Affinity / heavy-tail / CI / self-violation (legacy)"),
]


def cmd_task23():
    nb = load_nb()
    for old, new in _DEMOTE_HEADERS:
        for c in nb["cells"]:
            if c["cell_type"] != "markdown":
                continue
            src = "".join(c["source"])
            if old in src:
                c["source"] = src.replace(old, new).splitlines(keepends=True)

    end = next(i for i, c in enumerate(nb["cells"])
               if c["cell_type"] == "code"
               and "planning_safe_weight_range" in "".join(c["source"]))
    banner = md_cell(
        "# Appendix. Diagnostic background\n\n"
        "Below are the original Part A–E analyses (global conflict, null "
        "baseline, distribution, affinity, heavy-tail, self-violation). They "
        "are preserved verbatim for traceability but the main story now "
        "lives in Parts F–K above. Read these only when you need to verify "
        "background claims.\n"
    )
    if not any(c["cell_type"] == "markdown"
               and "Appendix. Diagnostic background" in "".join(c["source"])
               for c in nb["cells"][end + 1:end + 3]):
        _insert_after(nb, end, [banner])

    intro = nb["cells"][1]
    intro_src = "".join(intro["source"])
    note = ("\n\n> **Note (2026-05-12):** This notebook has been refactored "
            "around plan-centric directed transfer (Parts F–K). The original "
            "Part A–E analyses are now in the Appendix.\n")
    if "Note (2026-05-12)" not in intro_src:
        intro["source"] = (intro_src + note).splitlines(keepends=True)

    save_nb(nb)
    print("task23 applied")


CMDS["task23"] = cmd_task23


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    args = ap.parse_args()
    CMDS[args.cmd]()
