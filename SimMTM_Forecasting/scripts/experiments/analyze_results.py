#!/usr/bin/env python3
"""Plan v2 Task 6 — Statistical analysis of 5 loss x 5 seed sweep on ETTh1.

For each loss_mode, parse SimMTM_Forecasting/outputs/test_results/ETTh1/{loss_mode}_score.txt
(one line per finetune-test run), aggregate across seeds, then run paired t-test +
Wilcoxon signed-rank with Holm-Bonferroni correction over C(5,2)=10 pairwise comparisons.

Outputs:
    outputs/experiments/baseline_v1/results_table.md  (markdown summary)
    outputs/experiments/baseline_v1/results_table.csv (raw mean/std/p-values)
"""

import argparse
import os
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean, stdev

import numpy as np
from scipy import stats

DEFAULT_LOSS_MODES = ("mse", "sdsc", "hybrid", "zcr", "dilate")


def parse_score_file(path):
    """Return list of (mse, mae) tuples from a SimMTM `_score.txt`.

    Each line is formatted as `"{seq_len}->{pred_len}, {mse}, {mae}\n"`.
    """
    results = []
    if not Path(path).exists():
        return results
    with open(path) as f:
        for line in f:
            m = re.match(r"\s*(\d+)\s*->\s*(\d+)\s*,\s*([0-9.\-eE]+)\s*,\s*([0-9.\-eE]+)\s*$", line)
            if m:
                results.append((float(m.group(3)), float(m.group(4))))
    return results


def aggregate(score_dir, loss_modes):
    """Return {loss_mode: {'mse': [...], 'mae': [...]}}.

    score_dir is expected to be SimMTM_Forecasting/outputs/. Score files live
    at {score_dir}/{data}_{loss_mode}_score.txt OR
       {score_dir}/test_results/{data}/{loss_mode}_score.txt
    """
    out = {lm: {"mse": [], "mae": []} for lm in loss_modes}
    for lm in loss_modes:
        # Try both filename patterns used by the codebase.
        candidates = [
            Path(score_dir) / f"ETTh1_{lm}_score.txt",
            Path(score_dir) / "test_results" / "ETTh1" / f"{lm}_score.txt",
        ]
        for p in candidates:
            if p.exists():
                for mse_v, mae_v in parse_score_file(p):
                    out[lm]["mse"].append(mse_v)
                    out[lm]["mae"].append(mae_v)
    return out


def holm_bonferroni(pvals, alpha=0.05):
    """Sort p-values ascending, compare to alpha/(m-i+1). Returns array of corrected p-values."""
    pvals = np.asarray(pvals)
    m = len(pvals)
    order = np.argsort(pvals)
    corrected = np.empty(m)
    prev = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, pvals[idx] * (m - rank))
        corrected[idx] = max(prev, adj)  # enforce monotonic non-decreasing
        prev = corrected[idx]
    return corrected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--score-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "..", "outputs"),
        help="Directory containing ETTh1_{loss_mode}_score.txt files",
    )
    ap.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "experiments", "baseline_v1"),
    )
    ap.add_argument("--loss-modes", nargs="+", default=list(DEFAULT_LOSS_MODES))
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = aggregate(args.score_dir, args.loss_modes)

    # ----- per-loss summary
    rows = []
    for lm in args.loss_modes:
        m = data[lm]["mse"]
        a = data[lm]["mae"]
        if not m:
            rows.append((lm, 0, float("nan"), float("nan"), float("nan"), float("nan")))
            continue
        rows.append(
            (
                lm,
                len(m),
                mean(m),
                stdev(m) if len(m) > 1 else 0.0,
                mean(a),
                stdev(a) if len(a) > 1 else 0.0,
            )
        )

    # ----- pairwise tests (paired t + Wilcoxon)
    pairs = list(combinations(args.loss_modes, 2))
    pair_rows_mse = []
    pair_rows_mae = []
    for metric, pair_rows in [("mse", pair_rows_mse), ("mae", pair_rows_mae)]:
        ttest_ps = []
        wilcoxon_ps = []
        valid_pairs = []
        for a, b in pairs:
            xa = np.array(data[a][metric])
            xb = np.array(data[b][metric])
            if len(xa) == 0 or len(xb) == 0 or len(xa) != len(xb):
                ttest_ps.append(float("nan"))
                wilcoxon_ps.append(float("nan"))
                valid_pairs.append((a, b))
                continue
            t_stat, t_p = stats.ttest_rel(xa, xb)
            try:
                w_stat, w_p = stats.wilcoxon(xa, xb)
            except ValueError:
                w_p = float("nan")
            ttest_ps.append(t_p)
            wilcoxon_ps.append(w_p)
            valid_pairs.append((a, b))

        finite_t = [p for p in ttest_ps if not np.isnan(p)]
        finite_w = [p for p in wilcoxon_ps if not np.isnan(p)]
        if finite_t:
            holm_t = holm_bonferroni(finite_t, args.alpha)
        else:
            holm_t = []
        if finite_w:
            holm_w = holm_bonferroni(finite_w, args.alpha)
        else:
            holm_w = []
        it = iter(holm_t)
        iw = iter(holm_w)
        for (a, b), t_p, w_p in zip(valid_pairs, ttest_ps, wilcoxon_ps):
            t_p_corr = next(it) if not np.isnan(t_p) else float("nan")
            w_p_corr = next(iw) if not np.isnan(w_p) else float("nan")
            pair_rows.append((a, b, t_p, t_p_corr, w_p, w_p_corr))

    # ----- write CSV
    csv_path = out_dir / "results_table.csv"
    with open(csv_path, "w") as f:
        f.write("section,loss_mode_or_pair,n,mse_or_t_p,mae_or_w_p,extra_corrected\n")
        for lm, n, mse_m, mse_s, mae_m, mae_s in rows:
            f.write(f"summary,{lm},{n},mse={mse_m:.6f}±{mse_s:.6f},mae={mae_m:.6f}±{mae_s:.6f},\n")
        for a, b, t_p, t_pc, w_p, w_pc in pair_rows_mse:
            f.write(f"mse_test,{a}_vs_{b},,t_p={t_p:.4g},holm={t_pc:.4g},wilcoxon_p={w_p:.4g}/holm={w_pc:.4g}\n")
        for a, b, t_p, t_pc, w_p, w_pc in pair_rows_mae:
            f.write(f"mae_test,{a}_vs_{b},,t_p={t_p:.4g},holm={t_pc:.4g},wilcoxon_p={w_p:.4g}/holm={w_pc:.4g}\n")

    # ----- write Markdown
    md_path = out_dir / "results_table.md"
    with open(md_path, "w") as f:
        f.write("# ETTh1 pred_len=96 — 5-loss x N-seed sweep\n\n")
        f.write("## Per-loss summary (mean ± std over seeds)\n\n")
        f.write("| loss_mode | n_seeds | MSE mean±std | MAE mean±std |\n")
        f.write("|---|---|---|---|\n")
        for lm, n, mse_m, mse_s, mae_m, mae_s in rows:
            if n == 0:
                f.write(f"| {lm} | 0 | (no runs found) | |\n")
            else:
                f.write(f"| {lm} | {n} | {mse_m:.4f} ± {mse_s:.4f} | {mae_m:.4f} ± {mae_s:.4f} |\n")
        f.write("\n## Pairwise significance — MSE (Holm-Bonferroni corrected)\n\n")
        f.write("| pair | paired t p (corrected) | Wilcoxon p (corrected) |\n")
        f.write("|---|---|---|\n")
        for a, b, t_p, t_pc, w_p, w_pc in pair_rows_mse:
            star = "**" if not np.isnan(t_pc) and t_pc < args.alpha else ""
            f.write(f"| {a} vs {b} | {t_pc:.3g} {star} | {w_pc:.3g} |\n")
        f.write("\n## Pairwise significance — MAE (Holm-Bonferroni corrected)\n\n")
        f.write("| pair | paired t p (corrected) | Wilcoxon p (corrected) |\n")
        f.write("|---|---|---|\n")
        for a, b, t_p, t_pc, w_p, w_pc in pair_rows_mae:
            star = "**" if not np.isnan(t_pc) and t_pc < args.alpha else ""
            f.write(f"| {a} vs {b} | {t_pc:.3g} {star} | {w_pc:.3g} |\n")
        f.write("\nNote: '**' marks pairs significant at corrected alpha=" + str(args.alpha) + ".\n")
        f.write("\nN=5 seeds gives limited statistical power; consider N=10 for camera-ready.\n")

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
