#!/usr/bin/env python3
"""V1 metric evaluation — compute 8 metrics on the pre-registered pair set
and report per-family Spearman ρ + bootstrap CI + paired permutation tests
SDSC vs each baseline, with Benjamini-Hochberg FDR at q=0.05.

Inputs:
    paper_supplement/protocol/V1_pair_index.npz

Outputs:
    paper_supplement/protocol/V1_results.npz       (raw per-pair metric diffs)
    paper_supplement/protocol/V1_results_summary.md (human-readable table)

Per the V1 protocol, "predicted score" for each pair is
    s = metric(B, x) − metric(A, x)
positive s → A is closer to x → A is "more faithful" → predicted label = +1.
Ground-truth label conventions:
    +1: A more faithful
    -1: B more faithful
     0: NULL family
Spearman ρ between s (signed continuous) and ground-truth label gives the
metric's discriminative power for the pre-registered direction.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "SimMTM_Forecasting"))

from utils.sdsc_canonical import SignalDiceCanonical
from utils.baselines.zcr_diff import DiffZCRLoss
from utils.baselines.quantized_mse import one_bit_mse, two_bit_mu_law_mse
from utils.metrics import pearson_correlation, si_snr

PAIR_PATH = REPO_ROOT / "paper_supplement" / "protocol" / "V1_pair_index.npz"
RES_PATH = REPO_ROOT / "paper_supplement" / "protocol" / "V1_results.npz"
SUMMARY_PATH = REPO_ROOT / "paper_supplement" / "protocol" / "V1_results_summary.md"

N_BOOTSTRAP = 1000
N_PERMUTATION = 10000
FDR_Q = 0.05
SEED_BOOT = 7
SEED_PERM = 1729


# ─── Metric wrappers — all return tensor (scalar) of "distance" form ────
# smaller value = closer to x = more faithful
def m_mse(p, t): return torch.mean((p - t) ** 2)
def m_mae(p, t): return torch.mean(torch.abs(p - t))
_zcr = DiffZCRLoss(alpha=10.0)
def m_zcr(p, t): return _zcr(p, t)
def m_1bit(p, t): return one_bit_mse(p, t)
def m_2bit_mulaw(p, t): return two_bit_mu_law_mse(p, t)
def m_pcc(p, t):
    # pearson_correlation returns ∈ [-1, 1] (similarity). Convert to distance.
    return 1.0 - pearson_correlation(p.unsqueeze(0), t.unsqueeze(0))
def m_sisnr(p, t):
    # SI-SNR is in dB (higher=better). Negate for distance.
    return -si_snr(p.unsqueeze(0), t.unsqueeze(0))
_sdsc = SignalDiceCanonical(alpha=None)
def m_sdsc(p, t):
    return 1.0 - _sdsc(p.unsqueeze(0), t.unsqueeze(0))

METRICS = {
    "MSE": m_mse,
    "MAE": m_mae,
    "ZCR": m_zcr,
    "1-bit MSE": m_1bit,
    "2-bit μ-law MSE": m_2bit_mulaw,
    "PCC (1-r)": m_pcc,
    "SI-SNR (neg)": m_sisnr,
    "SDSC (1-s)": m_sdsc,
}


def compute_metric_diffs(x, A, B, metric_fn):
    """Per-pair s = metric(B,x) - metric(A,x), positive s → A more faithful."""
    n = x.shape[0]
    out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        xi = torch.from_numpy(x[i])
        Ai = torch.from_numpy(A[i])
        Bi = torch.from_numpy(B[i])
        ma = float(metric_fn(Ai, xi).item())
        mb = float(metric_fn(Bi, xi).item())
        out[i] = mb - ma
    return out


def discrim_score(s: np.ndarray, labels: np.ndarray) -> float:
    """Discrimination score appropriate to family structure.

    - mixed labels (>=2 unique non-null values): Spearman ρ in [-1, +1]
    - uniform non-null labels (e.g., all -1): accuracy in [0, 1] rescaled to
      [-1, +1] via 2*acc-1 so SDSC vs other comparisons stay on a common scale.
      Predicted label = sign(s). Accuracy counts pairs where sign(s) == label.
    - NULL families (label≡0): bias score = -|mean(s)| / std(s).  A symmetric
      metric (no bias) yields ≈ 0; biased metric yields negative score.
      Caps at [-1, +1] via tanh.
    """
    uniq = set(int(v) for v in np.unique(labels))
    if uniq == {0}:
        # NULL family
        m = float(np.mean(s)); sd = float(np.std(s) + 1e-8)
        bias = m / sd
        return float(-np.tanh(abs(bias)))
    if len(uniq - {0}) >= 2:
        # Mixed labels — Spearman ρ
        rho, _ = stats.spearmanr(s, labels)
        return float(rho if not np.isnan(rho) else 0.0)
    # Uniform single-class non-null label (e.g., all -1): accuracy
    target_sign = next(iter(uniq - {0}))
    # predicted_sign = sign(s); count matches with target_sign
    pred = np.sign(s)
    pred[pred == 0] = 1  # break ties as +1 (rare)
    acc = float(np.mean(pred == target_sign))
    return 2.0 * acc - 1.0  # rescale to [-1, +1]


def bootstrap_ci(s, labels, n_boot=N_BOOTSTRAP, rng=None):
    if rng is None:
        rng = np.random.default_rng(SEED_BOOT)
    n = len(s)
    scores = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        scores[b] = discrim_score(s[idx], labels[idx])
    lo = float(np.quantile(scores, 0.025))
    hi = float(np.quantile(scores, 0.975))
    return lo, hi


def paired_permutation_pvalue(s1, s2, labels, n_perm=N_PERMUTATION, rng=None):
    """Two-sided p-value for discrim_score(s1, labels) > discrim_score(s2, labels)
    via paired permutation: swap s1↔s2 assignment per pair."""
    if rng is None:
        rng = np.random.default_rng(SEED_PERM)
    obs = discrim_score(s1, labels) - discrim_score(s2, labels)
    count = 0
    for _ in range(n_perm):
        swap = rng.integers(0, 2, size=len(s1)).astype(bool)
        sa = np.where(swap, s2, s1); sb = np.where(swap, s1, s2)
        diff = discrim_score(sa, labels) - discrim_score(sb, labels)
        if abs(diff) >= abs(obs) - 1e-12:
            count += 1
    return (count + 1) / (n_perm + 1)


def bh_fdr(pvalues, q=FDR_Q):
    """Benjamini-Hochberg FDR. Returns boolean reject mask + adjusted p-values."""
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / (np.arange(1, n + 1))
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    reject = adj < q
    p_adj = np.zeros(n); p_adj[order] = adj
    rj = np.zeros(n, dtype=bool); rj[order] = reject
    return rj, p_adj


def main() -> int:
    if not PAIR_PATH.exists():
        print(f"ERROR: {PAIR_PATH} not found. Run generate_v1_pairs.py first.")
        return 1

    print(f"[V1 eval] Loading {PAIR_PATH}", flush=True)
    npz = np.load(PAIR_PATH, allow_pickle=True)
    x = npz["x"]; A = npz["A"]; B = npz["B"]
    labels = npz["labels"]
    families = list(npz["families"])
    F, N, T = x.shape
    print(f"[V1 eval] Pairs: {F} families × {N} pairs × T={T}", flush=True)

    # Compute s = metric(B,x) - metric(A,x) per family per metric
    diffs = np.zeros((len(METRICS), F, N), dtype=np.float32)
    for mi, (mname, mfn) in enumerate(METRICS.items()):
        print(f"[V1 eval] computing {mname}…", flush=True)
        for fi in range(F):
            diffs[mi, fi] = compute_metric_diffs(x[fi], A[fi], B[fi], mfn)

    # Per-family Spearman ρ + bootstrap CI + paired permutation vs SDSC
    metric_names = list(METRICS.keys())
    sdsc_idx = metric_names.index("SDSC (1-s)")

    rho_arr = np.zeros((len(METRICS), F))
    ci_lo = np.zeros_like(rho_arr); ci_hi = np.zeros_like(rho_arr)
    pval_vs_sdsc = np.ones((len(METRICS), F))
    rng_boot = np.random.default_rng(SEED_BOOT)
    rng_perm = np.random.default_rng(SEED_PERM)

    for mi, mname in enumerate(metric_names):
        for fi, fam in enumerate(families):
            lbl = labels[fi]
            s = diffs[mi, fi]
            rho_arr[mi, fi] = discrim_score(s, lbl)
            ci_lo[mi, fi], ci_hi[mi, fi] = bootstrap_ci(s, lbl, rng=rng_boot)
            if mi == sdsc_idx:
                pval_vs_sdsc[mi, fi] = 1.0
                continue
            s_sdsc = diffs[sdsc_idx, fi]
            pval_vs_sdsc[mi, fi] = paired_permutation_pvalue(s_sdsc, s, lbl, rng=rng_perm)

    # Benjamini-Hochberg FDR across (non-SDSC metric × family) pairs
    flat_p = pval_vs_sdsc[np.arange(len(METRICS)) != sdsc_idx].flatten()
    rj, p_adj = bh_fdr(flat_p, q=FDR_Q)
    pval_adj = np.zeros_like(pval_vs_sdsc)
    pval_adj[np.arange(len(METRICS)) != sdsc_idx] = p_adj.reshape(len(METRICS) - 1, F)
    reject_mask = np.zeros_like(pval_vs_sdsc, dtype=bool)
    reject_mask[np.arange(len(METRICS)) != sdsc_idx] = rj.reshape(len(METRICS) - 1, F)

    np.savez_compressed(
        RES_PATH,
        diffs=diffs,
        rho=rho_arr,
        ci_lo=ci_lo, ci_hi=ci_hi,
        pval_vs_sdsc=pval_vs_sdsc,
        pval_adj=pval_adj,
        reject_mask=reject_mask,
        metric_names=np.array(metric_names),
        families=np.array(families),
    )
    print(f"\n[V1 eval] Raw results → {RES_PATH}", flush=True)

    # ── Human-readable summary ──
    lines = []
    lines.append("# V1 Results — per-family Spearman ρ + paired permutation vs SDSC\n")
    lines.append(f"Pairs per family: {N}\n")
    lines.append(f"Bootstrap resamples: {N_BOOTSTRAP}\n")
    lines.append(f"Permutation iterations: {N_PERMUTATION}\n")
    lines.append(f"FDR correction (Benjamini-Hochberg q=): {FDR_Q}\n")
    lines.append("\n## Spearman ρ per (metric, family) — higher = better discrimination\n")
    header = "| metric | " + " | ".join(families) + " |"
    sep = "|" + "---|" * (len(families) + 1)
    lines.append(header); lines.append(sep)
    for mi, mname in enumerate(metric_names):
        row = f"| {mname} | "
        for fi, fam in enumerate(families):
            row += f"{rho_arr[mi, fi]:+.3f} | "
        lines.append(row.rstrip())

    lines.append("\n## Bootstrap 95% CI on ρ\n")
    lines.append(header); lines.append(sep)
    for mi, mname in enumerate(metric_names):
        row = f"| {mname} | "
        for fi, fam in enumerate(families):
            row += f"[{ci_lo[mi, fi]:+.2f}, {ci_hi[mi, fi]:+.2f}] | "
        lines.append(row.rstrip())

    lines.append("\n## Paired permutation p-value vs SDSC (BH-adjusted)\n")
    lines.append("`*` = reject H0 (SDSC ≠ baseline) at FDR q=0.05\n")
    lines.append(header); lines.append(sep)
    for mi, mname in enumerate(metric_names):
        if mi == sdsc_idx:
            row = f"| {mname} | " + " | ".join(["—"] * len(families)) + " |"
        else:
            row = f"| {mname} | "
            for fi, fam in enumerate(families):
                p = pval_adj[mi, fi]; rj = reject_mask[mi, fi]
                row += f"{p:.3g}{'*' if rj else ''} | "
            row = row.rstrip()
        lines.append(row)

    lines.append("\n## Family (g) AC-7 fork criterion check\n")
    fi_g = families.index("g") if "g" in families else None
    if fi_g is not None:
        sdsc_g = rho_arr[sdsc_idx, fi_g]
        zcr_i = metric_names.index("ZCR")
        bit1_i = metric_names.index("1-bit MSE")
        bit2_i = metric_names.index("2-bit μ-law MSE")
        lines.append(f"- ρ_SDSC(g) = {sdsc_g:+.3f}")
        lines.append(f"- ρ_ZCR(g)  = {rho_arr[zcr_i, fi_g]:+.3f}")
        lines.append(f"- ρ_1bit(g) = {rho_arr[bit1_i, fi_g]:+.3f}")
        lines.append(f"- ρ_2bit(g) = {rho_arr[bit2_i, fi_g]:+.3f}")
        lines.append(f"- Pre-registered ordering: SDSC > 2-bit > 1-bit > ZCR")
        ok_zcr = sdsc_g > rho_arr[zcr_i, fi_g] and ci_lo[sdsc_idx, fi_g] > ci_hi[zcr_i, fi_g]
        ok_1bit = sdsc_g >= rho_arr[bit1_i, fi_g]
        lines.append(f"- AC-7(B) SDSC > ZCR with non-overlapping CI: **{'PASS' if ok_zcr else 'FAIL'}**")
        lines.append(f"- AC-7(C) SDSC ≥ 1-bit MSE: **{'PASS' if ok_1bit else 'FAIL'}**")

    SUMMARY_PATH.write_text("\n".join(lines))
    print(f"[V1 eval] Summary → {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
