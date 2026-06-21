#!/usr/bin/env python3
"""Multi-axis analyzer for the multi-dataset × multi-model × multi-loss sweep.

Parses score files at:
    outputs/test_results/{dataset}/{dataset}_{loss}_score.txt

The format may be 3-col (legacy: "{seq}->{pred}, mse, mae")
or 4-col (current: "{seq}->{pred}, mse, mae, sdsc"). Both are tolerated.

Important caveat: the score file does not know which MODEL produced each row.
exp.test() in this codebase appends one line per finetune-test invocation
without recording the model name. To disambiguate by model, we re-parse the
sweep status TSV (multi_sweep_v1/run_status.tsv) which records (dataset, model,
loss) DONE entries in execution order, and align that order with the score-file
rows. This is fragile if status / score files diverge; we warn when counts
mismatch.

Outputs:
    outputs/experiments/multi_sweep_v1/results_table.md
    outputs/experiments/multi_sweep_v1/results_table.csv

For each dataset, a per-model table of 8 losses × {MSE, MAE, SDSC}.
With multi-seed runs, also includes mean ± std per (model, loss) and paired
t-test / Wilcoxon with Holm-Bonferroni across loss-pairs (within the same
model+dataset).
"""

import argparse
import re
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

try:
    from scipy import stats
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from datasets_config import DATASETS, LOSS_MODES, MODELS  # noqa: E402

REPO_ROOT = SCRIPT_DIR.parent.parent


def parse_score_file(path):
    """Return [(mse, mae, sdsc, softdtw, dilate), ...] tuples; missing cols are NaN."""
    rows = []
    if not path.exists():
        return rows
    rx3 = re.compile(r"\s*\d+\s*->\s*\d+\s*,\s*([0-9.\-eE]+)\s*,\s*([0-9.\-eE]+)\s*$")
    rx4 = re.compile(r"\s*\d+\s*->\s*\d+\s*,\s*([0-9.\-eE]+)\s*,\s*([0-9.\-eE]+)\s*,\s*([0-9.\-eE]+)\s*$")
    rx6 = re.compile(r"\s*\d+\s*->\s*\d+\s*,\s*([0-9.\-eE]+)\s*,\s*([0-9.\-eE]+)\s*,\s*([0-9.\-eE]+)\s*,\s*([0-9.\-eE]+)\s*,\s*([0-9.\-eE]+)\s*$")
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            m6 = rx6.match(line)
            if m6:
                rows.append((float(m6.group(1)), float(m6.group(2)),
                             float(m6.group(3)), float(m6.group(4)), float(m6.group(5))))
                continue
            m4 = rx4.match(line)
            if m4:
                rows.append((float(m4.group(1)), float(m4.group(2)),
                             float(m4.group(3)), float("nan"), float("nan")))
                continue
            m3 = rx3.match(line)
            if m3:
                rows.append((float(m3.group(1)), float(m3.group(2)),
                             float("nan"), float("nan"), float("nan")))
    return rows


def parse_run_status(status_file):
    """Yield (timestamp, dataset, model, loss, status) tuples in file order."""
    if not status_file.exists():
        return []
    out = []
    with open(status_file) as f:
        next(f, None)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 5:
                out.append(tuple(parts[:5]))
    return out


def holm_bonferroni(pvals):
    """Holm-Bonferroni correction (monotone non-decreasing)."""
    pvals = np.asarray(pvals)
    m = len(pvals)
    order = np.argsort(pvals)
    out = np.empty(m)
    prev = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, pvals[idx] * (m - rank))
        out[idx] = max(prev, adj)
        prev = out[idx]
    return out


def aggregate(score_dir, sweep_dir):
    """Return {(dataset, model, loss): {'mse': [...], 'mae': [...], 'sdsc': [...]}}."""
    status = parse_run_status(sweep_dir / "run_status.tsv")
    # Build expected order per (dataset, loss): list of models in execution order.
    order_per_dl = defaultdict(list)
    for ts, dataset, model, loss, st in status:
        if st == "DONE":
            order_per_dl[(dataset, loss)].append(model)

    data = defaultdict(lambda: {"mse": [], "mae": [], "sdsc": [], "softdtw": [], "dilate": []})
    for dataset in DATASETS:
        for loss in LOSS_MODES:
            path = score_dir / "test_results" / DATASETS[dataset]["data"] / f"{DATASETS[dataset]['data']}_{loss}_score.txt"
            rows = parse_score_file(path)
            ordered_models = order_per_dl.get((dataset, loss), [])
            n_score = len(rows)
            n_models = len(ordered_models)
            if n_score == 0:
                continue
            if n_score != n_models:
                if n_score < n_models:
                    ordered_models = ordered_models[-n_score:]
                else:
                    rows = rows[-n_models:]
                    n_score = len(rows)
            for (mse, mae, sdsc, sdtw, dlt), model in zip(rows, ordered_models):
                key = (dataset, model, loss)
                data[key]["mse"].append(mse)
                data[key]["mae"].append(mae)
                data[key]["sdsc"].append(sdsc)
                data[key]["softdtw"].append(sdtw)
                data[key]["dilate"].append(dlt)
    return data


def summarize(data):
    """Return {(dataset, model, loss): {'mse_mean', 'mse_std', ...}}."""
    rows = {}
    for key, d in data.items():
        n = len(d["mse"])
        if n == 0:
            continue
        s = {"n": n}
        for metric in ["mse", "mae", "sdsc", "softdtw", "dilate"]:
            arr = np.array([v for v in d[metric] if not np.isnan(v)])
            s[f"{metric}_mean"] = float(arr.mean()) if arr.size else float("nan")
            s[f"{metric}_std"] = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
            s[f"{metric}_n_valid"] = int(arr.size)
        rows[key] = s
    return rows


def write_markdown(out_path, summary, data):
    """Per-dataset tables, per-model loss × {MSE, MAE, SDSC}."""
    with open(out_path, "w") as f:
        f.write("# Multi-dataset × multi-model × multi-loss sweep — Results\n\n")
        f.write("Datasets: " + ", ".join(DATASETS.keys()) + "\n")
        f.write("Models:   " + ", ".join(MODELS) + "\n")
        f.write("Losses:   " + ", ".join(LOSS_MODES) + "\n")
        f.write("Metrics:  MSE, MAE, SDSC (downstream test set)\n\n")

        for dataset in DATASETS:
            data_name = DATASETS[dataset]["data"]
            f.write(f"---\n\n## {dataset}\n\n")
            for model in MODELS:
                rows_present = [loss for loss in LOSS_MODES if (dataset, model, loss) in summary]
                if not rows_present:
                    f.write(f"### {model}\n\nNo runs yet.\n\n")
                    continue
                metric_vals = {m: [summary[(dataset, model, l)][f"{m}_mean"] for l in rows_present]
                               for m in ["mse", "mae", "sdsc", "softdtw", "dilate"]}
                # lower is better: mse, mae, softdtw, dilate. higher is better: sdsc.
                def safe_argmin(vals):
                    return int(np.nanargmin(vals)) if not all(np.isnan(vals)) else None
                def safe_argmax(vals):
                    return int(np.nanargmax(vals)) if not all(np.isnan(vals)) else None
                best_idx = {
                    "mse":     safe_argmin(metric_vals["mse"]),
                    "mae":     safe_argmin(metric_vals["mae"]),
                    "sdsc":    safe_argmax(metric_vals["sdsc"]),
                    "softdtw": safe_argmin(metric_vals["softdtw"]),
                    "dilate":  safe_argmin(metric_vals["dilate"]),
                }

                f.write(f"### {model}\n\n")
                f.write("| loss | n | MSE (↓) | MAE (↓) | SDSC (↑) | SoftDTW (↓) | DILATE (↓) |\n")
                f.write("|---|---|---|---|---|---|---|\n")
                for i, loss in enumerate(rows_present):
                    s = summary[(dataset, model, loss)]
                    def cell(metric, idx, best, fmt="{:.4f}"):
                        m = s[f"{metric}_mean"]
                        std = s[f"{metric}_std"]
                        if np.isnan(m):
                            return "n/a"
                        text = fmt.format(m) + (f" ± {std:.4f}" if s["n"] > 1 else "")
                        if best is not None and idx == best:
                            text = f"**{text}**"
                        return text
                    f.write(f"| {loss} | {s['n']} | "
                            f"{cell('mse', i, best_idx['mse'])} | "
                            f"{cell('mae', i, best_idx['mae'])} | "
                            f"{cell('sdsc', i, best_idx['sdsc'])} | "
                            f"{cell('softdtw', i, best_idx['softdtw'])} | "
                            f"{cell('dilate', i, best_idx['dilate'])} |\n")
                f.write("\n")

                # Holm-Bonferroni pairwise per metric (only when multi-seed)
                if HAS_SCIPY and any(summary[(dataset, model, l)]["n"] >= 2 for l in rows_present):
                    f.write("Pairwise paired t-test (Holm-Bonferroni, α=0.05) — only with N≥2 per loss:\n")
                    for metric in ["mse", "mae", "sdsc", "softdtw", "dilate"]:
                        pairs = list(combinations(rows_present, 2))
                        pvals, valid = [], []
                        for a, b in pairs:
                            xa = [v for v in data[(dataset, model, a)][metric] if not np.isnan(v)]
                            xb = [v for v in data[(dataset, model, b)][metric] if not np.isnan(v)]
                            if len(xa) < 2 or len(xb) < 2 or len(xa) != len(xb):
                                continue
                            _, p = stats.ttest_rel(xa, xb)
                            pvals.append(p); valid.append((a, b))
                        if not pvals:
                            continue
                        holm = holm_bonferroni(pvals)
                        f.write(f"\n#### {metric.upper()}\n\n| pair | raw p | Holm-corrected p | sig α=0.05 |\n|---|---|---|---|\n")
                        for (a, b), p, pc in zip(valid, pvals, holm):
                            sig = "**YES**" if pc < 0.05 else "no"
                            f.write(f"| {a} vs {b} | {p:.4g} | {pc:.4g} | {sig} |\n")
                        f.write("\n")
    print(f"Wrote {out_path}")


def write_csv(out_path, summary):
    with open(out_path, "w") as f:
        f.write("dataset,model,loss,n,"
                "mse_mean,mse_std,mae_mean,mae_std,sdsc_mean,sdsc_std,"
                "softdtw_mean,softdtw_std,dilate_mean,dilate_std\n")
        for (dataset, model, loss), s in summary.items():
            f.write(f"{dataset},{model},{loss},{s['n']},"
                    f"{s['mse_mean']},{s['mse_std']},"
                    f"{s['mae_mean']},{s['mae_std']},"
                    f"{s['sdsc_mean']},{s['sdsc_std']},"
                    f"{s['softdtw_mean']},{s['softdtw_std']},"
                    f"{s['dilate_mean']},{s['dilate_std']}\n")
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-dir", default=str(REPO_ROOT / "outputs"))
    ap.add_argument("--sweep-dir", default=str(REPO_ROOT / "outputs" / "experiments" / "multi_sweep_v1"))
    args = ap.parse_args()

    data = aggregate(Path(args.score_dir), Path(args.sweep_dir))
    summary = summarize(data)
    out_dir = Path(args.sweep_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_markdown(out_dir / "results_table.md", summary, data)
    write_csv(out_dir / "results_table.csv", summary)


if __name__ == "__main__":
    main()
