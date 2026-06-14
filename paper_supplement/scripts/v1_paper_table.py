#!/usr/bin/env python3
"""Render V1 results into a paper-ready Markdown table (for direct copy/paste).

Reads V1_results.npz and produces a compact Section-5 results table with:
  - rows: 8 metrics
  - cols: 10 families (a-j), plus an "AC-7 fork check (g)" column

For each cell, prints: discrim_score [CI_lo, CI_hi]  (paired p* if reject FDR)
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RES_PATH = REPO_ROOT / "paper_supplement" / "protocol" / "V1_results.npz"
OUT_PATH = REPO_ROOT / "paper_supplement" / "protocol" / "V1_paper_table.md"


def main() -> int:
    if not RES_PATH.exists():
        print(f"Run evaluate_v1_metrics.py first ({RES_PATH} missing).")
        return 1
    z = np.load(RES_PATH, allow_pickle=True)
    rho = z["rho"]                # (M, F) — discrim_score in [-1, +1]
    ci_lo = z["ci_lo"]; ci_hi = z["ci_hi"]
    pval_adj = z["pval_adj"]
    reject = z["reject_mask"]
    metric_names = list(z["metric_names"])
    families = list(z["families"])

    sdsc_idx = metric_names.index("SDSC (1-s)")
    zcr_idx = metric_names.index("ZCR")
    b1_idx = metric_names.index("1-bit MSE")
    b2_idx = metric_names.index("2-bit μ-law MSE")

    lines = []
    lines.append("# V1 Paper Table (Section 5)\n")
    lines.append("`discrim_score` ∈ [−1, +1]: for win-families = 2·accuracy − 1; "
                 "for NULL families = −tanh(|mean(s)|/std(s)); for mixed (j) = Spearman ρ.\n")
    lines.append("Higher = better discrimination of pre-registered structural label.\n")
    lines.append("`*` = paired permutation vs SDSC rejects equivalence at FDR q=0.05.\n")

    header = "| metric | " + " | ".join(families) + " |"
    sep = "|" + "---|" * (len(families) + 1)
    lines.append(header); lines.append(sep)

    for mi, mname in enumerate(metric_names):
        row = f"| **{mname}** | "
        for fi, fam in enumerate(families):
            score = rho[mi, fi]
            lo, hi = ci_lo[mi, fi], ci_hi[mi, fi]
            mark = "*" if (mi != sdsc_idx and reject[mi, fi]) else ""
            row += f"{score:+.2f} [{lo:+.2f},{hi:+.2f}]{mark} | "
        lines.append(row.rstrip())

    # AC-7 fork verdict for family g
    fi_g = families.index("g") if "g" in families else None
    if fi_g is not None:
        lines.append("\n## AC-7 fork criterion check on family (g) — sign-preserving structural damage\n")
        sdsc_g = rho[sdsc_idx, fi_g]
        zcr_g = rho[zcr_idx, fi_g]
        b1_g = rho[b1_idx, fi_g]
        b2_g = rho[b2_idx, fi_g]
        sdsc_lo = ci_lo[sdsc_idx, fi_g]; sdsc_hi = ci_hi[sdsc_idx, fi_g]
        zcr_lo = ci_lo[zcr_idx, fi_g]; zcr_hi = ci_hi[zcr_idx, fi_g]
        b1_lo = ci_lo[b1_idx, fi_g]; b1_hi = ci_hi[b1_idx, fi_g]

        lines.append("| metric | discrim_score (g) | 95% CI |")
        lines.append("|---|---|---|")
        lines.append(f"| SDSC | {sdsc_g:+.3f} | [{sdsc_lo:+.3f}, {sdsc_hi:+.3f}] |")
        lines.append(f"| 2-bit μ-law MSE | {b2_g:+.3f} | [{ci_lo[b2_idx,fi_g]:+.3f}, {ci_hi[b2_idx,fi_g]:+.3f}] |")
        lines.append(f"| 1-bit MSE | {b1_g:+.3f} | [{b1_lo:+.3f}, {b1_hi:+.3f}] |")
        lines.append(f"| ZCR | {zcr_g:+.3f} | [{zcr_lo:+.3f}, {zcr_hi:+.3f}] |")
        lines.append("")
        lines.append("**Pre-registered ordering**: SDSC > 2-bit > 1-bit > ZCR\n")

        # AC-7 fork conditions
        cond_B = (sdsc_g > zcr_g) and (sdsc_lo > zcr_hi)
        cond_C = sdsc_g >= b1_g
        cond_A = "TBD (multi-family check)"
        lines.append(f"- AC-7 (B) SDSC > ZCR (non-overlapping 95% CI): **{'PASS' if cond_B else 'FAIL'}**")
        lines.append(f"- AC-7 (C) SDSC ≥ 1-bit MSE: **{'PASS' if cond_C else 'FAIL'}**")
        lines.append(f"- AC-7 (A) multi-family criterion to be evaluated separately.")
        lines.append("")
        verdict = "AAAI-clean (if A also passes)" if (cond_B and cond_C) else (
            "AAAI-honest" if cond_B or cond_C else "TMLR pivot recommended"
        )
        lines.append(f"**Preliminary verdict (g-only)**: {verdict}")

    OUT_PATH.write_text("\n".join(lines))
    print(f"V1 paper table → {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
