#!/usr/bin/env python3
"""AAAI27 protocol v2 unified analyzer (C-4 + C-5 across TF-C, TS2Vec, SimMTM-Cls).

Reads:
  SimMTM_Classification/outputs/v2_sweep/{backbone}/{dt}/seed{N}/{c4_result.json, c5_*_result.json}
  SimMTM_Classification/outputs/classification_sweep/*.json  (SimMTM-Cls v1 results)

Computes:
  AC-CL2-2 (C-4 SDSC > ZCR generalization)
  AC-CL2-3 (C-5 loss-neutrality in-domain ≤ 2%)
  AC-CL2-4 (cross-domain TOST ±3% accuracy, BH-FDR q=0.05)

Outputs:
  paper_supplement/protocol/v2_results.npz       (raw + per-cell stats)
  paper_supplement/protocol/v2_results_summary.md (human-readable verdicts)
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
V2_DIR = REPO_ROOT / "SimMTM_Classification" / "outputs" / "v2_sweep"
V3_DIR = REPO_ROOT / "SimMTM_Classification" / "outputs" / "v3_gpt4ts_sweep"
V1_DIR = REPO_ROOT / "SimMTM_Classification" / "outputs" / "classification_sweep"
SEEDCELLS = REPO_ROOT / "paper_supplement" / "protocol" / "classification_seedcells_v2.json"
RES_NPZ = REPO_ROOT / "paper_supplement" / "protocol" / "v2_results.npz"
SUMMARY_MD = REPO_ROOT / "paper_supplement" / "protocol" / "v2_results_summary.md"

ZCR_CATASTROPHIC_PCT = 5.0
LOSS_NEUTRAL_PCT = 2.0
TOST_MARGIN_ACC = 3.0
FDR_Q = 0.05
N_PERM = 10000
SEED_PERM = 1729


def collect_v2():
    """Returns:
       c4[(backbone, dt_id, seed)] = {metric_name: value}
       c5[(backbone, dt_id, seed, loss)] = accuracy
    Includes v2_sweep (TFC/TS2Vec) + v3_gpt4ts_sweep (GPT4TS).
    """
    c4 = {}
    c5 = {}

    # V3 GPT4TS layout: v3_gpt4ts_sweep/{dt_id}/seed{N}/c4_result.json (no backbone subdir)
    if V3_DIR.exists():
        for c4_json in V3_DIR.rglob("c4_result.json"):
            try:
                d = json.loads(c4_json.read_text())
                backbone = d["backbone"]  # "GPT4TS" from JSON content
                parts = c4_json.parts
                dt_id = parts[-3]
                seed = int(parts[-2].replace("seed", ""))
                c4[(backbone, dt_id, seed)] = d.get("metrics", {})
            except Exception:
                continue
        for c5_json in V3_DIR.rglob("c5_*_result.json"):
            try:
                d = json.loads(c5_json.read_text())
                backbone = d["backbone"]
                loss = d["recon_loss"]
                parts = c5_json.parts
                dt_id = parts[-3]
                seed = int(parts[-2].replace("seed", ""))
                c5[(backbone, dt_id, seed, loss)] = d.get("accuracy", -1.0)
            except Exception:
                continue

    if not V2_DIR.exists():
        return c4, c5
    for c4_json in V2_DIR.rglob("c4_result.json"):
        try:
            d = json.loads(c4_json.read_text())
            backbone = d["backbone"]
            # dt_id from path: v2_sweep/{backbone}/{dt_id}/seed{N}/c4_result.json
            parts = c4_json.parts
            dt_id = parts[-3]
            seed = int(parts[-2].replace("seed", ""))
            c4[(backbone, dt_id, seed)] = d.get("metrics", {})
        except Exception:
            continue
    for c5_json in V2_DIR.rglob("c5_*_result.json"):
        try:
            d = json.loads(c5_json.read_text())
            backbone = d["backbone"]
            loss = d["recon_loss"]
            parts = c5_json.parts
            dt_id = parts[-3]
            seed = int(parts[-2].replace("seed", ""))
            c5[(backbone, dt_id, seed, loss)] = d.get("accuracy", -1.0)
        except Exception:
            continue
    return c4, c5


def collect_v1_simmtm_cls():
    """SimMTM-Cls v1 sweep writes per-cell .log files (not JSON). Parse the
    final 'EP20 - Best Testing: Acc=...' line from each log.
    Returns c5_simmtm[(dt_id, seed, loss)] = accuracy (in [0, 1])."""
    out = {}
    if not V1_DIR.exists():
        return out
    fn_re = re.compile(r"^(.+)_seed(\d+)\.log$")
    cell_re = re.compile(r"^(?:indomain|xdomain)_([a-z]+)_(.+)$")
    acc_re = re.compile(r"Best Testing:\s*Acc\s*=\s*([\d.]+)")
    for lpath in V1_DIR.glob("*.log"):
        m = fn_re.match(lpath.name)
        if not m:
            continue
        cell_id, seed = m.group(1), int(m.group(2))
        cm = cell_re.match(cell_id)
        if not cm:
            continue
        loss = cm.group(1)
        target_part = cm.group(2)
        if cell_id.startswith("indomain_"):
            dt_id = f"in_{target_part}"
        else:
            if "__" not in target_part:
                continue
            src, tgt = target_part.split("__", 1)
            dt_id = f"xd_{src}_{tgt}"
        try:
            text = lpath.read_text()
            best_match = None
            for line in text.splitlines():
                am = acc_re.search(line)
                if am:
                    best_match = am
            if best_match:
                acc_pct = float(best_match.group(1))  # log uses percent (e.g. 94.58)
                out[(dt_id, seed, loss)] = acc_pct / 100.0
        except Exception:
            continue
    return out


def paired_perm(x1, x2, n_perm=N_PERM, rng=None):
    if len(x1) < 2:
        return 1.0
    if rng is None:
        rng = np.random.default_rng(SEED_PERM)
    obs = float(np.mean(x1) - np.mean(x2))
    cnt = 0
    for _ in range(n_perm):
        swap = rng.integers(0, 2, size=len(x1)).astype(bool)
        a = np.where(swap, x2, x1); b = np.where(swap, x1, x2)
        if abs(float(np.mean(a) - np.mean(b))) >= abs(obs) - 1e-12:
            cnt += 1
    return (cnt + 1) / (n_perm + 1)


def tost_pvalue(x1, x2, margin):
    if len(x1) < 2 or len(x2) < 2:
        return 1.0
    d = np.asarray(x1) - np.asarray(x2)
    m = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    if se == 0:
        return 0.0 if abs(m) < margin else 1.0
    df = len(d) - 1
    t_lo = (m + margin) / se
    t_hi = (m - margin) / se
    p_lo = 1.0 - stats.t.cdf(t_lo, df)
    p_hi = float(stats.t.cdf(t_hi, df))
    return float(max(p_lo, p_hi))


def bh_fdr(pvals, q=FDR_Q):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = p[order] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    p_adj = np.zeros(n); p_adj[order] = adj
    rej = np.zeros(n, dtype=bool); rej[order] = adj < q
    return rej, p_adj


def main() -> int:
    c4, c5 = collect_v2()
    c5_simmtm = collect_v1_simmtm_cls()
    spec = json.loads(SEEDCELLS.read_text())
    dt_specs = spec["dataset_types"]
    seeds = spec["config"]["seeds"]
    losses = spec["config"]["losses_c5"]

    print(f"[v2 analyze] c4 results: {len(c4)}", flush=True)
    print(f"[v2 analyze] c5 v2 results: {len(c5)}", flush=True)
    print(f"[v2 analyze] c5 SimMTM-Cls v1 results: {len(c5_simmtm)}", flush=True)

    # ── AC-CL2-2: C-4 SDSC vs ZCR per (backbone, dt) ──
    # Pre-registered direction: SDSC > ZCR_soft as discrimination of
    # "the encoder that produces structurally faithful reconstructions".
    # Since higher SDSC = better (similarity), and higher ZCR_soft = worse
    # (distance), our test is whether SDSC rankings of encoders agree
    # MORE with reconstruction-fidelity ground truth than ZCR rankings.
    # Operationalize: per (backbone, dt) pair, accumulate SDSC and ZCR_soft
    # over seeds; compare paired across backbones.
    lines = []
    lines.append("# AAAI27 protocol v2 cross-backbone results\n")
    lines.append("- ZCR catastrophic threshold (in-domain): MSE acc − ZCR acc > 5%")
    lines.append("- Loss-neutrality threshold (in-domain): |SDSC acc − MSE acc| ≤ 2%")
    lines.append("- Cross-domain TOST margin: ±3% accuracy, BH-FDR q=0.05\n")

    # Combined view of C-5: backbone → dt → loss → list of accuracies across seeds
    c5_all = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for (backbone, dt_id, seed, loss), acc in c5.items():
        c5_all[backbone][dt_id][loss].append(acc)
    for (dt_id, seed, loss), acc in c5_simmtm.items():
        c5_all["SimMTM-Cls"][dt_id][loss].append(acc)

    # ── AC-CL2-3 (loss-neutrality in-domain) ──
    lines.append("## AC-CL2-3: Loss-neutrality on in-domain (|SDSC − MSE| ≤ 2%)\n")
    lines.append("| backbone | dataset | MSE mean | SDSC mean | ZCR mean | |SDSC−MSE| | ZCR drop | neutral? | catastrophic? |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    neutral_count = 0
    neutral_total = 0
    catastrophic_count = 0
    catastrophic_total = 0
    for backbone in sorted(c5_all):
        for dt in sorted(c5_all[backbone]):
            if not dt.startswith("in_"):
                continue
            row = c5_all[backbone][dt]
            def _mean(lst): return float(np.mean(lst)) if lst else None
            mse_m = _mean(row.get("mse", []))
            sdsc_m = _mean(row.get("sdsc", []))
            zcr_m = _mean(row.get("zcr", []))
            if mse_m is None or sdsc_m is None:
                continue
            diff = abs(sdsc_m - mse_m) * 100
            neutral = diff <= LOSS_NEUTRAL_PCT
            neutral_count += int(neutral); neutral_total += 1
            if zcr_m is not None:
                drop = (mse_m - zcr_m) * 100
                cat = drop > ZCR_CATASTROPHIC_PCT
                catastrophic_count += int(cat); catastrophic_total += 1
                drop_str = f"{drop:+.2f}%"
                cat_str = "✓" if cat else " "
            else:
                drop_str = "—"; cat_str = "—"
            lines.append(
                f"| {backbone} | {dt} | {mse_m:.4f} | {sdsc_m:.4f} "
                f"| {(zcr_m if zcr_m is not None else 0):.4f} | {diff:.2f}% "
                f"| {drop_str} | {'✓' if neutral else ' '} | {cat_str} |"
            )
    lines.append(
        f"\n**Loss-neutrality**: {neutral_count}/{neutral_total} in-domain cells pass ≤2% threshold."
    )
    lines.append(
        f"**ZCR catastrophic (AC-CL2-2)**: {catastrophic_count}/{catastrophic_total} in-domain cells show > 5% drop."
    )

    # ── AC-CL2-4 (cross-domain TOST + paired permutation) ──
    lines.append("\n## AC-CL2-4: Cross-domain TOST (±3% acc, BH-FDR q=0.05)\n")
    lines.append("| backbone | source→target | MSE mean | SDSC mean | TOST p (raw) | TOST p (BH) | equivalent? |")
    lines.append("|---|---|---|---|---|---|---|")
    tost_rows = []
    for backbone in sorted(c5_all):
        for dt in sorted(c5_all[backbone]):
            if not dt.startswith("xd_"):
                continue
            row = c5_all[backbone][dt]
            mse_a = row.get("mse", []); sdsc_a = row.get("sdsc", [])
            if not mse_a or not sdsc_a or len(mse_a) != len(sdsc_a):
                continue
            margin = TOST_MARGIN_ACC / 100.0
            p_raw = tost_pvalue(mse_a, sdsc_a, margin)
            tost_rows.append({
                "backbone": backbone, "dt": dt,
                "mse_mean": float(np.mean(mse_a)),
                "sdsc_mean": float(np.mean(sdsc_a)),
                "p_raw": p_raw,
            })
    if tost_rows:
        rej, p_adj = bh_fdr([r["p_raw"] for r in tost_rows], q=FDR_Q)
        for r, rj, pa in zip(tost_rows, rej, p_adj):
            r["p_adj"] = float(pa); r["equivalent"] = bool(rj)
            lines.append(
                f"| {r['backbone']} | {r['dt']} | {r['mse_mean']:.4f} | {r['sdsc_mean']:.4f} "
                f"| {r['p_raw']:.4g} | {r['p_adj']:.4g} | {'✓' if r['equivalent'] else ' '} |"
            )
        equiv = sum(1 for r in tost_rows if r["equivalent"])
        lines.append(
            f"\n**Cross-domain TOST**: {equiv}/{len(tost_rows)} pairs equivalent at ±3% (BH-FDR q=0.05)."
        )
    else:
        lines.append("(no cross-domain data yet)")

    # ── AC-CL2-2 (C-4 SDSC vs ZCR_soft, backbone discrimination) ──
    lines.append("\n## AC-CL2-2: C-4 reconstruction metrics across backbones\n")
    if c4:
        # Aggregate per (backbone, dt) — average across seeds
        agg = defaultdict(lambda: defaultdict(list))  # (backbone, dt) → metric → list
        for (backbone, dt_id, seed), metrics in c4.items():
            for k, v in metrics.items():
                agg[(backbone, dt_id)][k].append(v)
        # Per dt, show backbone ranking by SDSC vs ZCR
        for dt in sorted({k[1] for k in agg}):
            lines.append(f"\n### dataset-type: {dt}\n")
            lines.append("| backbone | SDSC ↑ | ZCR_soft ↓ | MSE ↓ | PCC ↑ |")
            lines.append("|---|---|---|---|---|")
            for backbone in sorted({k[0] for k in agg if k[1] == dt}):
                m = agg[(backbone, dt)]
                lines.append(
                    f"| {backbone} | {np.mean(m.get('SDSC', [0])):.4f} "
                    f"| {np.mean(m.get('ZCR_soft', [0])):.4f} "
                    f"| {np.mean(m.get('MSE', [0])):.4f} "
                    f"| {np.mean(m.get('PCC', [0])):.4f} |"
                )
    else:
        lines.append("(no C-4 results yet)")

    # Save raw + summary
    np.savez_compressed(
        RES_NPZ,
        c4=np.array([json.dumps({"key": list(k), "metrics": v}) for k, v in c4.items()]),
        c5=np.array([json.dumps({"key": list(k), "acc": v}) for k, v in c5.items()]),
        c5_simmtm=np.array([json.dumps({"key": list(k), "acc": v}) for k, v in c5_simmtm.items()]),
        tost=np.array([json.dumps(r) for r in tost_rows]) if tost_rows else np.array([]),
    )
    SUMMARY_MD.write_text("\n".join(lines))
    print(f"[v2 analyze] summary → {SUMMARY_MD}", flush=True)
    print(f"[v2 analyze] raw → {RES_NPZ}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
