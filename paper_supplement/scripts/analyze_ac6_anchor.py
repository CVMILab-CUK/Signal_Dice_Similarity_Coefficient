#!/usr/bin/env python3
"""AC-6 60-run anchor analysis — TOST equivalence + BH FDR.

Parses per-cell logs in outputs/experiments/ac6_anchor/ to recover
unambiguous (dataset, model, loss, seed) → {test_mse, test_mae, test_sdsc}
mapping. Then:

  1. Per cell {(loss, dataset, backbone)}, compute mean ± std across 5 seeds.
  2. For each (dataset, backbone) and each baseline loss vs SDSC / MSE,
     run TOST equivalence test at ±0.5% MSE margin (post-M3 pilot tightened).
  3. Multiple-comparison correction via Benjamini-Hochberg FDR (q=0.05).
  4. Report which loss-pair comparisons are statistically equivalent.

Output:
  paper_supplement/protocol/AC6_results.npz       (raw + group stats)
  paper_supplement/protocol/AC6_results_summary.md (human-readable)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
AC6_DIR = REPO_ROOT / "SimMTM_Forecasting" / "outputs" / "experiments" / "ac6_anchor"
RES_PATH = REPO_ROOT / "paper_supplement" / "protocol" / "AC6_results.npz"
SUMMARY_PATH = REPO_ROOT / "paper_supplement" / "protocol" / "AC6_results_summary.md"
SEEDCELLS_PATH = REPO_ROOT / "paper_supplement" / "protocol" / "ac6_seedcells.json"

# TOST equivalence margin: ±0.5% MSE relative to mean (tightened from ±1.4%)
TOST_MARGIN_REL = 0.005
FDR_Q = 0.05

# Log line patterns
LOG_RE_TEST = re.compile(
    r"96->96,\s*mse:([0-9.]+),\s*mae:([0-9.]+),\s*sdsc:([0-9.]+)"
)
LOG_RE_FILENAME = re.compile(
    r"([A-Za-z0-9]+)_([A-Za-z0-9]+)_([a-z]+)_seed(\d+)\.log$"
)


def parse_cell_log(log_path: Path):
    """Return dict {mse, mae, sdsc} from the last matching line."""
    last = None
    try:
        for line in log_path.read_text().splitlines():
            m = LOG_RE_TEST.search(line)
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
    }


def collect_results():
    rows = []
    for log_path in sorted(AC6_DIR.glob("*_seed*.log")):
        fn_match = LOG_RE_FILENAME.search(log_path.name)
        if not fn_match:
            continue
        dataset, model, loss, seed_str = fn_match.groups()
        seed = int(seed_str)
        metrics = parse_cell_log(log_path)
        if metrics is None:
            continue
        rows.append({
            "dataset": dataset, "model": model, "loss": loss, "seed": seed,
            **metrics,
            "log": log_path.name,
        })
    return rows


def tost_pvalue(x1, x2, margin_abs):
    """Two One-Sided Test for equivalence within ±margin_abs (paired t)."""
    if len(x1) < 2 or len(x2) < 2:
        return 1.0
    d = np.asarray(x1) - np.asarray(x2)
    mean_d = d.mean()
    se = d.std(ddof=1) / np.sqrt(len(d))
    if se == 0:
        return 0.0 if abs(mean_d) < margin_abs else 1.0
    df = len(d) - 1
    t_lower = (mean_d + margin_abs) / se
    t_upper = (mean_d - margin_abs) / se
    p_lower = 1 - stats.t.cdf(t_lower, df)
    p_upper = stats.t.cdf(t_upper, df)
    return max(p_lower, p_upper)


def bh_fdr(pvalues, q=FDR_Q):
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = p[order] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    p_adj = np.zeros(n); p_adj[order] = adj
    reject = np.zeros(n, dtype=bool); reject[order] = adj < q
    return reject, p_adj


def main() -> int:
    if not AC6_DIR.exists():
        print(f"ERROR: AC-6 sweep dir not found: {AC6_DIR}")
        return 1
    rows = collect_results()
    print(f"[AC6 analysis] parsed {len(rows)} cell logs", flush=True)
    if not rows:
        print("[AC6 analysis] no DONE cells yet; rerun after sweep completes")
        return 1

    seedcells = json.loads(SEEDCELLS_PATH.read_text()) if SEEDCELLS_PATH.exists() else None

    # Group: (loss, dataset, model) → list[mse over 5 seeds]
    groups = {}
    for r in rows:
        key = (r["loss"], r["dataset"], r["model"])
        groups.setdefault(key, []).append(r)

    # Per-cell mean/std table
    cell_stats = {}
    for key, group in groups.items():
        ms = np.array([r["mse"] for r in group])
        cell_stats[key] = {
            "n": len(ms),
            "mean": float(ms.mean()),
            "std": float(ms.std(ddof=1)) if len(ms) > 1 else 0.0,
            "cv_pct": float(100 * ms.std(ddof=1) / ms.mean()) if len(ms) > 1 else 0.0,
            "mse_per_seed": ms.tolist(),
        }

    # TOST equivalence tests per (dataset, model) comparing loss pairs
    losses = sorted({k[0] for k in cell_stats})
    datasets = sorted({k[1] for k in cell_stats})
    models = sorted({k[2] for k in cell_stats})

    tost_rows = []
    for dataset in datasets:
        for model in models:
            for i, l1 in enumerate(losses):
                for l2 in losses[i + 1:]:
                    k1 = (l1, dataset, model); k2 = (l2, dataset, model)
                    if k1 not in cell_stats or k2 not in cell_stats:
                        continue
                    g1 = [r["mse"] for r in groups[k1]]
                    g2 = [r["mse"] for r in groups[k2]]
                    if len(g1) != len(g2):
                        continue
                    pooled_mean = (np.mean(g1) + np.mean(g2)) / 2
                    margin_abs = TOST_MARGIN_REL * pooled_mean
                    p = tost_pvalue(g1, g2, margin_abs)
                    tost_rows.append({
                        "dataset": dataset, "model": model,
                        "loss_a": l1, "loss_b": l2,
                        "p": float(p),
                        "margin_abs": float(margin_abs),
                        "diff": float(np.mean(g1) - np.mean(g2)),
                    })

    # FDR over all TOST p-values
    if tost_rows:
        pvals = [r["p"] for r in tost_rows]
        reject, p_adj = bh_fdr(pvals)
        for r, rj, pa in zip(tost_rows, reject, p_adj):
            r["p_adj"] = float(pa)
            r["reject_h0"] = bool(rj)  # reject H0: NOT equivalent → ARE equivalent

    np.savez_compressed(
        RES_PATH,
        rows=np.array([json.dumps(r) for r in rows]),
        cell_stats=np.array([json.dumps({"key": list(k), **v}) for k, v in cell_stats.items()]),
        tost=np.array([json.dumps(t) for t in tost_rows]),
    )
    print(f"\n[AC6 analysis] raw → {RES_PATH}", flush=True)

    # Summary
    lines = []
    lines.append("# AC-6 60-Run Anchor Analysis — Loss Neutrality TOST Test\n")
    lines.append(f"- equivalence margin: ±{TOST_MARGIN_REL*100:.1f}% MSE (post-M3 tightened)")
    lines.append(f"- multiple-comparison correction: Benjamini-Hochberg FDR at q={FDR_Q}")
    lines.append(f"- cells parsed: {len(groups)} groups, {len(rows)} total runs\n")

    lines.append("## Per-cell summary (loss × dataset × backbone): mean ± std across 5 seeds\n")
    lines.append("| loss | dataset | backbone | n | mean MSE | std | CV (%) |")
    lines.append("|---|---|---|---|---|---|---|")
    for (loss, ds, mdl), s in sorted(cell_stats.items()):
        lines.append(
            f"| {loss} | {ds} | {mdl} | {s['n']} | {s['mean']:.4f} | {s['std']:.4f} | {s['cv_pct']:.3f}% |"
        )

    lines.append("\n## TOST equivalence test (loss pairs, per dataset×backbone)\n")
    lines.append("`✓` = TOST rejects H0 (NOT equivalent) → losses ARE statistically equivalent within ±0.5% MSE\n")
    lines.append("| dataset | backbone | loss A | loss B | diff | p (raw) | p (BH) | equivalent? |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for t in sorted(tost_rows, key=lambda r: (r["dataset"], r["model"], r["loss_a"], r["loss_b"])):
        mark = "✓" if t.get("reject_h0", False) else " "
        lines.append(
            f"| {t['dataset']} | {t['model']} | {t['loss_a']} | {t['loss_b']} | "
            f"{t['diff']:+.4f} | {t['p']:.4g} | {t.get('p_adj', float('nan')):.4g} | {mark} |"
        )

    # Headline: fraction of equivalent pairs
    if tost_rows:
        equiv_frac = np.mean([r.get("reject_h0", False) for r in tost_rows])
        lines.append(f"\n**Headline:** {equiv_frac:.0%} of loss-pair comparisons are TOST-equivalent within ±0.5% MSE (FDR-corrected).")

    SUMMARY_PATH.write_text("\n".join(lines))
    print(f"[AC6 analysis] summary → {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
