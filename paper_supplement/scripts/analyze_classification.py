#!/usr/bin/env python3
"""AAAI27 Classification Confirmatory analyzer (Plan B++ AC-CL-2/3/4).

Reads outputs/classification_sweep/{cell_id}_seed{seed}.json files (TFC
finetune output) and computes:

  AC-CL-2: ZCR catastrophic threshold — acc drop > 5% vs MSE on ≥ 2/3
           in-domain datasets.
  AC-CL-3: Loss-neutrality threshold — |acc(SDSC) − acc(MSE)| ≤ 1% on all
           3 in-domain datasets.
  AC-CL-4: Cross-domain TOST equivalence at ±3% accuracy margin via paired
           permutation test SDSC vs MSE, BH-FDR at q=0.05.

Outputs:
  paper_supplement/protocol/classification_results.npz
  paper_supplement/protocol/classification_results_summary.md
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
SEEDCELLS = REPO_ROOT / "paper_supplement" / "protocol" / "classification_seedcells.json"
SWEEP_DIR = REPO_ROOT / "SimMTM_Classification" / "outputs" / "classification_sweep"
RES_PATH = REPO_ROOT / "paper_supplement" / "protocol" / "classification_results.npz"
SUMMARY_PATH = REPO_ROOT / "paper_supplement" / "protocol" / "classification_results_summary.md"

ZCR_CATASTROPHIC_PCT = 5.0
LOSS_NEUTRAL_PCT = 1.0
TOST_MARGIN_ACC = 3.0
FDR_Q = 0.05
N_PERMUTATION = 10000
SEED_PERM = 1729


def collect_results():
    """Walk SWEEP_DIR/*.json, returns {cell_id: {seed: accuracy}}."""
    out = defaultdict(dict)
    spec = json.loads(SEEDCELLS.read_text())
    valid_ids = {c["id"] for c in spec["in_domain_cells"] + spec["cross_domain_cells"]}
    if not SWEEP_DIR.exists():
        return out, spec
    for jpath in sorted(SWEEP_DIR.glob("*.json")):
        # filename pattern: {cell_id}_seed{seed}.json
        stem = jpath.stem
        if "_seed" not in stem:
            continue
        cell_id, _, seed_str = stem.rpartition("_seed")
        if cell_id not in valid_ids:
            continue
        try:
            seed = int(seed_str)
            data = json.loads(jpath.read_text())
            # TFC writes various result fields; pick accuracy if present, else top1
            acc = float(data.get("accuracy", data.get("acc", data.get("top1", -1.0))))
            if acc < 0:
                continue
            out[cell_id][seed] = acc
        except Exception:
            continue
    return out, spec


def paired_perm_pval(x1, x2, n_perm=N_PERMUTATION, rng=None):
    if len(x1) < 2:
        return 1.0
    if rng is None:
        rng = np.random.default_rng(SEED_PERM)
    obs = float(np.mean(x1) - np.mean(x2))
    count = 0
    for _ in range(n_perm):
        swap = rng.integers(0, 2, size=len(x1)).astype(bool)
        a = np.where(swap, x2, x1)
        b = np.where(swap, x1, x2)
        diff = float(np.mean(a) - np.mean(b))
        if abs(diff) >= abs(obs) - 1e-12:
            count += 1
    return (count + 1) / (n_perm + 1)


def tost_pvalue(x1, x2, margin):
    """TOST equivalence test — accuracy units (percentage points)."""
    if len(x1) < 2 or len(x2) < 2:
        return 1.0
    d = np.asarray(x1) - np.asarray(x2)
    mean_d = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    if se == 0:
        return 0.0 if abs(mean_d) < margin else 1.0
    df = len(d) - 1
    t_lower = (mean_d + margin) / se
    t_upper = (mean_d - margin) / se
    p_lower = 1.0 - stats.t.cdf(t_lower, df)
    p_upper = float(stats.t.cdf(t_upper, df))
    return float(max(p_lower, p_upper))


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
    by_cell, spec = collect_results()
    if not by_cell:
        print(f"ERROR: no results found in {SWEEP_DIR}. Run sweep first.")
        return 1

    # Group cells by (kind, target, loss) — kind ∈ {in_domain, xdomain_{source}}
    grouped = defaultdict(dict)  # (kind, target, loss) -> {seed: acc}
    for cell_dict in spec["in_domain_cells"]:
        cid = cell_dict["id"]
        if cid in by_cell:
            grouped[("indomain", cell_dict["target"], cell_dict["loss"])] = by_cell[cid]
    for cell_dict in spec["cross_domain_cells"]:
        cid = cell_dict["id"]
        if cid in by_cell:
            key = (f"xdomain_{cell_dict['source']}", cell_dict["target"],
                   cell_dict["loss"])
            grouped[key] = by_cell[cid]

    # ── AC-CL-2 + AC-CL-3 (in-domain) ──
    in_targets = spec["config"]["in_domain_targets"]
    in_table = []
    for tgt in in_targets:
        row = {"target": tgt}
        for loss in ("mse", "sdsc", "zcr"):
            seeds_accs = grouped.get(("indomain", tgt, loss), {})
            accs = list(seeds_accs.values())
            if accs:
                row[f"{loss}_mean"] = float(np.mean(accs))
                row[f"{loss}_std"] = float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0
                row[f"{loss}_n"] = len(accs)
            else:
                row[f"{loss}_mean"] = None
                row[f"{loss}_std"] = None
                row[f"{loss}_n"] = 0
        in_table.append(row)

    # AC-CL-2 ZCR catastrophic check
    zcr_catastrophic_count = 0
    for row in in_table:
        if row.get("zcr_mean") is None or row.get("mse_mean") is None:
            continue
        drop = row["mse_mean"] - row["zcr_mean"]
        row["zcr_drop_pct"] = drop * 100.0
        row["zcr_catastrophic"] = drop * 100.0 > ZCR_CATASTROPHIC_PCT
        if row["zcr_catastrophic"]:
            zcr_catastrophic_count += 1
    ac_cl_2_pass = zcr_catastrophic_count >= 2

    # AC-CL-3 loss-neutrality check
    loss_neutral_all = True
    for row in in_table:
        if row.get("sdsc_mean") is None or row.get("mse_mean") is None:
            loss_neutral_all = False
            continue
        diff = abs(row["sdsc_mean"] - row["mse_mean"]) * 100.0
        row["sdsc_mse_diff_pct"] = diff
        row["loss_neutral"] = diff <= LOSS_NEUTRAL_PCT
        if not row["loss_neutral"]:
            loss_neutral_all = False
    ac_cl_3_pass = loss_neutral_all

    # ── AC-CL-4 (cross-domain TOST + paired permutation) ──
    rng_perm = np.random.default_rng(SEED_PERM)
    xd_table = []
    tost_pvals = []
    for pair in spec["config"]["cross_domain_pairs"]:
        src = pair["source"]; tgt = pair["target"]
        row = {"source": src, "target": tgt}
        mse_accs = list(grouped.get((f"xdomain_{src}", tgt, "mse"), {}).values())
        sdsc_accs = list(grouped.get((f"xdomain_{src}", tgt, "sdsc"), {}).values())
        zcr_accs = list(grouped.get((f"xdomain_{src}", tgt, "zcr"), {}).values())
        for nm, accs in (("mse", mse_accs), ("sdsc", sdsc_accs), ("zcr", zcr_accs)):
            if accs:
                row[f"{nm}_mean"] = float(np.mean(accs))
                row[f"{nm}_std"] = float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0
            else:
                row[f"{nm}_mean"] = None; row[f"{nm}_std"] = None
        if mse_accs and sdsc_accs and len(mse_accs) == len(sdsc_accs):
            # TOST: accuracy units * 100 to align with margin in pct
            margin_pct = TOST_MARGIN_ACC / 100.0
            row["tost_p"] = tost_pvalue(mse_accs, sdsc_accs, margin_pct)
            row["paired_perm_p"] = paired_perm_pval(mse_accs, sdsc_accs, rng=rng_perm)
            tost_pvals.append(row["tost_p"])
        xd_table.append(row)

    if tost_pvals:
        reject, p_adj = bh_fdr(tost_pvals, q=FDR_Q)
        for r, rj, pa in zip(
            [r for r in xd_table if "tost_p" in r], reject, p_adj
        ):
            r["tost_p_adj"] = float(pa); r["tost_equivalent"] = bool(rj)
    ac_cl_4_xd_equiv = sum(1 for r in xd_table if r.get("tost_equivalent", False))

    summary = {
        "ac_cl_2_zcr_catastrophic": ac_cl_2_pass,
        "ac_cl_2_count": zcr_catastrophic_count,
        "ac_cl_3_loss_neutrality": ac_cl_3_pass,
        "ac_cl_4_xd_equivalent_pairs": ac_cl_4_xd_equiv,
        "ac_cl_4_total_pairs": len(tost_pvals),
    }
    np.savez_compressed(
        RES_PATH,
        in_table=np.array([json.dumps(r) for r in in_table]),
        xd_table=np.array([json.dumps(r) for r in xd_table]),
        summary=np.array([json.dumps(summary)]),
    )

    # ── Markdown summary ──
    lines = []
    lines.append("# Classification Confirmatory Results — Plan B++ AC checks\n")
    lines.append(f"- ZCR catastrophic threshold: > {ZCR_CATASTROPHIC_PCT}% drop vs MSE")
    lines.append(f"- Loss-neutrality threshold: ≤ {LOSS_NEUTRAL_PCT}% |SDSC − MSE|")
    lines.append(f"- Cross-domain TOST margin: ±{TOST_MARGIN_ACC}% accuracy, BH-FDR q={FDR_Q}")
    lines.append("")

    lines.append("## In-domain accuracy (mean ± std across seeds)\n")
    lines.append("| target | MSE | SDSC | ZCR | ZCR drop | |SDSC−MSE| | C4 catastrophic | C2 neutral |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for row in in_table:
        def _fmt(loss):
            m = row.get(f"{loss}_mean"); s = row.get(f"{loss}_std")
            if m is None: return "n/a"
            return f"{m:.4f}±{s:.4f}" if s is not None else f"{m:.4f}"
        cat = "✓" if row.get("zcr_catastrophic", False) else " "
        ntr = "✓" if row.get("loss_neutral", False) else " "
        drop = f"{row.get('zcr_drop_pct', 0):+.2f}%" if "zcr_drop_pct" in row else "—"
        diff = f"{row.get('sdsc_mse_diff_pct', 0):.2f}%" if "sdsc_mse_diff_pct" in row else "—"
        lines.append(
            f"| {row['target']} | {_fmt('mse')} | {_fmt('sdsc')} | {_fmt('zcr')} "
            f"| {drop} | {diff} | {cat} | {ntr} |"
        )

    lines.append(
        f"\n**AC-CL-2 (ZCR catastrophic)**: "
        f"{zcr_catastrophic_count}/3 in-domain datasets show > {ZCR_CATASTROPHIC_PCT}% drop "
        f"→ **{'PASS' if ac_cl_2_pass else 'FAIL'}** (threshold ≥ 2)."
    )
    lines.append(
        f"\n**AC-CL-3 (Loss-neutrality)**: "
        f"All 3 in-domain |SDSC − MSE| ≤ {LOSS_NEUTRAL_PCT}% "
        f"→ **{'PASS' if ac_cl_3_pass else 'FAIL'}**."
    )

    lines.append("\n## Cross-domain transfer (AC-CL-4)\n")
    lines.append("| source | target | MSE | SDSC | ZCR | TOST p | TOST p (BH) | equivalent? |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in xd_table:
        def _fmt2(nm):
            m = r.get(f"{nm}_mean"); s = r.get(f"{nm}_std")
            if m is None: return "n/a"
            return f"{m:.4f}±{s:.4f}" if s is not None else f"{m:.4f}"
        eq = "✓" if r.get("tost_equivalent", False) else " "
        tp = f"{r.get('tost_p', float('nan')):.4g}" if "tost_p" in r else "—"
        ta = f"{r.get('tost_p_adj', float('nan')):.4g}" if "tost_p_adj" in r else "—"
        lines.append(
            f"| {r['source']} | {r['target']} | {_fmt2('mse')} | {_fmt2('sdsc')} "
            f"| {_fmt2('zcr')} | {tp} | {ta} | {eq} |"
        )
    lines.append(
        f"\n**AC-CL-4 (Cross-domain TOST)**: "
        f"{ac_cl_4_xd_equiv}/{len(tost_pvals)} pairs TOST-equivalent at "
        f"±{TOST_MARGIN_ACC}% accuracy (BH-FDR q={FDR_Q})."
    )

    SUMMARY_PATH.write_text("\n".join(lines))
    print(f"[classification analyze] summary → {SUMMARY_PATH}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
