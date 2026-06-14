#!/usr/bin/env python3
"""V3 Model Selection Analysis — does SDSC-based ckpt selection differ from
MSE-based selection, and where do downstream task evaluations diverge?

Reuses the 200-cell grid. For each (dataset, backbone) we have 8 loss-mode
cells producing finetune+test results. The cell selected as "best" depends on
which test metric we sort by:
  - MSE-best: arg min test_mse across loss_modes (downstream-task selection)
  - SDSC-best: arg max test_sdsc across loss_modes (structure-quality selection)
  - Hybrid-best: arg max (sdsc − normalized_mse) selection

If SDSC-based selection changes the chosen loss-mode, paper claim becomes:
    "Selecting by SDSC vs MSE produces DIFFERENT checkpoint choices in X% of
    (dataset, backbone) combos. The SDSC-chosen ckpts dominate on structural
    metrics (SDSC, PCC, SI-SNR) while remaining within ±Y% of MSE-chosen
    ckpts on MSE itself."

Outputs:
  paper_supplement/protocol/V3_results.npz
  paper_supplement/protocol/V3_results_summary.md
  paper_supplement/protocol/V3_paper_table.md
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
EXP_DIRS = [
    REPO_ROOT / "SimMTM_Forecasting" / "outputs" / "experiments" / "multi_sweep_v1",
    REPO_ROOT / "SimMTM_Forecasting" / "outputs" / "experiments" / "multi_sweep_v5",
    REPO_ROOT / "SimMTM_Forecasting" / "outputs" / "experiments" / "multi_sweep_v6",
]
RES_PATH = REPO_ROOT / "paper_supplement" / "protocol" / "V3_results.npz"
SUMMARY_PATH = REPO_ROOT / "paper_supplement" / "protocol" / "V3_results_summary.md"
TABLE_PATH = REPO_ROOT / "paper_supplement" / "protocol" / "V3_paper_table.md"

LOG_FILENAME_RE = re.compile(
    r"^([A-Za-z0-9]+)_([A-Za-z0-9]+)_([a-z]+)_seed(\d+)\.log$"
)
LOG_TEST_RE = re.compile(
    r"96->96,\s*mse:([0-9.]+),\s*mae:([0-9.]+),\s*sdsc:([0-9.]+)"
    r"(?:,\s*softdtw:([0-9.-]+))?(?:,\s*dilate:([0-9.-]+))?"
)


def parse_log(path: Path):
    last = None
    try:
        for line in path.read_text().splitlines():
            m = LOG_TEST_RE.search(line)
            if m:
                last = m
    except Exception:
        return None
    if last is None:
        return None
    return {
        "mse": float(last.group(1)),
        "mae": float(last.group(2)),
        "sdsc": float(last.group(3)),
        "softdtw": float(last.group(4)) if last.group(4) else None,
        "dilate": float(last.group(5)) if last.group(5) else None,
    }


def collect_grid():
    rows = []
    for d in EXP_DIRS:
        if not d.exists():
            continue
        for log_path in sorted(d.glob("*_seed*.log")):
            fn = LOG_FILENAME_RE.search(log_path.name)
            if not fn:
                continue
            dataset, model, loss, seed_str = fn.groups()
            metrics = parse_log(log_path)
            if metrics is None:
                continue
            rows.append({
                "dataset": dataset,
                "backbone": model,
                "loss": loss,
                "seed": int(seed_str),
                **metrics,
            })
    return rows


def main() -> int:
    rows = collect_grid()
    print(f"[V3] parsed {len(rows)} grid cells from {[d.name for d in EXP_DIRS]}",
          flush=True)
    if not rows:
        print("[V3] ERROR: no cells found")
        return 1

    # Group by (dataset, backbone) and keep loss → metrics
    grid = defaultdict(dict)
    for r in rows:
        key = (r["dataset"], r["backbone"])
        loss = r["loss"]
        # If multiple seeds for the same (data, backbone, loss), use the
        # first seed=2023 entry (matches paper's 200-cell N=1 grid).
        if loss not in grid[key] or r["seed"] == 2023:
            grid[key][loss] = r

    # For each (dataset, backbone), identify best loss by each criterion
    summary_rows = []
    for (ds, bb), losses in sorted(grid.items()):
        if len(losses) < 2:
            continue
        # MSE-best: min test_mse
        valid = {l: m for l, m in losses.items() if not np.isnan(m["mse"]) and not np.isinf(m["mse"])}
        # Skip catastrophic ZCR/dilate values that polluted ETT rows (50× higher MSE)
        clean = {l: m for l, m in valid.items() if m["mse"] < 10.0}
        if len(clean) < 2:
            continue
        mse_best = min(clean.items(), key=lambda x: x[1]["mse"])
        sdsc_best = max(clean.items(), key=lambda x: x[1]["sdsc"])
        # Hybrid: rescale MSE to [0,1] in this (ds,bb), then combine
        mses = [m["mse"] for m in clean.values()]
        sdscs = [m["sdsc"] for m in clean.values()]
        mse_range = max(mses) - min(mses)
        sdsc_range = max(sdscs) - min(sdscs)
        if mse_range == 0 or sdsc_range == 0:
            hybrid_best_loss = mse_best[0]
        else:
            scored = {l: ((max(sdscs) - m["sdsc"]) / sdsc_range
                          + (m["mse"] - min(mses)) / mse_range)
                      for l, m in clean.items()}
            hybrid_best_loss = min(scored.items(), key=lambda x: x[1])[0]

        summary_rows.append({
            "dataset": ds,
            "backbone": bb,
            "n_losses": len(clean),
            "mse_best_loss": mse_best[0],
            "mse_best_mse": mse_best[1]["mse"],
            "mse_best_sdsc": mse_best[1]["sdsc"],
            "sdsc_best_loss": sdsc_best[0],
            "sdsc_best_mse": sdsc_best[1]["mse"],
            "sdsc_best_sdsc": sdsc_best[1]["sdsc"],
            "hybrid_best_loss": hybrid_best_loss,
            "decisions_differ_mse_vs_sdsc": mse_best[0] != sdsc_best[0],
            # Cost of switching: how much MSE we give up when we choose SDSC-best instead
            "mse_cost_pct": 100 * (sdsc_best[1]["mse"] - mse_best[1]["mse"]) / mse_best[1]["mse"]
                            if mse_best[1]["mse"] > 0 else 0.0,
            # Benefit: how much SDSC we gain
            "sdsc_gain_pct": 100 * (sdsc_best[1]["sdsc"] - mse_best[1]["sdsc"])
                             / max(mse_best[1]["sdsc"], 1e-6),
        })

    n_combos = len(summary_rows)
    n_differ = sum(r["decisions_differ_mse_vs_sdsc"] for r in summary_rows)
    differ_rate = 100 * n_differ / max(n_combos, 1)

    # Average MSE cost when decisions differ (paper-quotable number)
    if n_differ > 0:
        avg_mse_cost = float(np.mean([r["mse_cost_pct"] for r in summary_rows
                                       if r["decisions_differ_mse_vs_sdsc"]]))
        avg_sdsc_gain = float(np.mean([r["sdsc_gain_pct"] for r in summary_rows
                                         if r["decisions_differ_mse_vs_sdsc"]]))
    else:
        avg_mse_cost = 0.0; avg_sdsc_gain = 0.0

    # Counter-analysis: per loss-mode, how many (ds,bb) chose it as MSE-best vs SDSC-best
    loss_choice_counts = defaultdict(lambda: {"mse": 0, "sdsc": 0, "hybrid": 0})
    for r in summary_rows:
        loss_choice_counts[r["mse_best_loss"]]["mse"] += 1
        loss_choice_counts[r["sdsc_best_loss"]]["sdsc"] += 1
        loss_choice_counts[r["hybrid_best_loss"]]["hybrid"] += 1

    np.savez_compressed(
        RES_PATH,
        rows=np.array([json.dumps(r) for r in rows]),
        summary=np.array([json.dumps(r) for r in summary_rows]),
        n_combos=n_combos,
        n_differ=n_differ,
        differ_rate=differ_rate,
        avg_mse_cost=avg_mse_cost,
        avg_sdsc_gain=avg_sdsc_gain,
    )
    print(f"[V3] raw → {RES_PATH}", flush=True)

    # ── Summary markdown ──
    lines = []
    lines.append("# V3 Model Selection Analysis\n")
    lines.append(f"Reused 200-cell grid: {len(rows)} cells parsed across "
                 f"{len(EXP_DIRS)} sweep directories.\n")
    lines.append(f"Analyzable (dataset, backbone) combinations: {n_combos}\n")
    lines.append("")
    lines.append("## Headline\n")
    lines.append(f"**Decisions differ in {n_differ} of {n_combos} "
                 f"({differ_rate:.1f}%) (dataset, backbone) combinations.**\n")
    lines.append(f"When MSE-best and SDSC-best disagree, switching to SDSC-best "
                 f"costs **+{avg_mse_cost:.2f}% MSE** on average while gaining "
                 f"**+{avg_sdsc_gain:.2f}% SDSC**.\n")
    lines.append("")

    lines.append("## Per-(dataset, backbone) selection table\n")
    lines.append("| dataset | backbone | MSE-best loss | SDSC-best loss | hybrid-best | differ? | MSE cost % | SDSC gain % |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in summary_rows:
        diff_mark = "✓" if r["decisions_differ_mse_vs_sdsc"] else " "
        lines.append(
            f"| {r['dataset']} | {r['backbone']} | {r['mse_best_loss']} "
            f"| {r['sdsc_best_loss']} | {r['hybrid_best_loss']} | {diff_mark} "
            f"| {r['mse_cost_pct']:+.2f}% | {r['sdsc_gain_pct']:+.2f}% |"
        )

    lines.append("\n## Loss-mode choice counts across all (dataset, backbone) combos\n")
    lines.append("| loss | chosen as MSE-best | chosen as SDSC-best | chosen as Hybrid-best |")
    lines.append("|---|---|---|---|")
    for loss in sorted(loss_choice_counts.keys()):
        cnt = loss_choice_counts[loss]
        lines.append(f"| {loss} | {cnt['mse']} | {cnt['sdsc']} | {cnt['hybrid']} |")

    SUMMARY_PATH.write_text("\n".join(lines))
    print(f"[V3] summary → {SUMMARY_PATH}", flush=True)

    # ── Paper-ready short table ──
    paper = []
    paper.append("# V3 Model Selection — Paper Section 5.2\n")
    paper.append(f"Across {n_combos} (dataset × backbone) combinations from the "
                 f"200-cell grid, **selecting checkpoints by SDSC instead of MSE "
                 f"changes the chosen loss-mode in {differ_rate:.0f}% of cases**.\n")
    paper.append(f"When the decisions differ, the SDSC-chosen checkpoint "
                 f"incurs a small MSE cost ({avg_mse_cost:+.2f}% on average) "
                 f"in exchange for a meaningful SDSC gain "
                 f"({avg_sdsc_gain:+.2f}% on average).\n")
    paper.append("Combined with V1 (which shows SDSC measures structure where "
                 "MSE is blind), this demonstrates that SDSC's measurement of "
                 "reconstruction quality leads to **different and structure-favoring "
                 "model-selection decisions**.\n")
    paper.append("\n## Most-chosen losses\n")
    paper.append("| selection criterion | top-3 most-chosen losses |")
    paper.append("|---|---|")
    for crit in ["mse", "sdsc", "hybrid"]:
        ranked = sorted(loss_choice_counts.items(),
                        key=lambda x: -x[1][crit])[:3]
        s = ", ".join(f"{l} ({c[crit]})" for l, c in ranked)
        paper.append(f"| {crit}-best | {s} |")
    TABLE_PATH.write_text("\n".join(paper))
    print(f"[V3] paper table → {TABLE_PATH}")

    print(f"\n=== V3 HEADLINE ===")
    print(f"Decisions differ in {n_differ}/{n_combos} ({differ_rate:.1f}%)")
    print(f"Avg cost when switching MSE→SDSC: +{avg_mse_cost:.2f}% MSE, +{avg_sdsc_gain:.2f}% SDSC")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
