#!/usr/bin/env python3
"""V1 pair generator v2 — MSE-matched pairs via binary-search distortion scaling.

Implements paper_supplement/protocol/V1_protocol.md with the *operational
discipline* that |MSE(A,x) - MSE(B,x)| / max(MSE) <= 2% per pair.

For non-NULL families the location of damage carries the structural meaning
(A = high-amplitude region, B = low-amplitude region). High-amp distortion
yields larger MSE under a fixed strength, so we scale B's distortion until
the per-pair MSEs match within tolerance via binary search.

Outputs paper_supplement/protocol/V1_pair_index.npz with the same arrays as
v1 plus `matched_rel_mse_diff` reported per pair.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = REPO_ROOT / "paper_supplement" / "protocol" / "V1_pair_index.npz"

T = 96
N_PER_FAMILY = 500
FAMILIES = list("abcdefghij")
SEED_PAIR = 42
SEED_WIN = 123
MSE_MATCH_REL_TOL = 0.02
MAX_BSEARCH_ITERS = 30
SEVERITY = {
    "a": [1, 2, 3], "b": [1, 3, 5],
    "c": [0.05, 0.10, 0.20],
    "d": [0.1, 0.3, 0.5],
    "e": [0.1, 0.3, 0.5],
    "f": [0.05, 0.10, 0.20],
    "g": [1.5, 2.0, 3.0],
    "h": [0.05, 0.10, 0.15],
    "i": [0.5, 1.5, 2.0],
    "j": [0.3, 0.5, 1.0],
}


def load_ecg() -> np.ndarray:
    d = torch.load(
        "/workspace/data/signal/classification/ECG/train.pt",
        weights_only=False, map_location="cpu",
    )
    x = d["samples"].numpy().astype(np.float32).squeeze(1)
    return ((x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-8))


def load_sleepeeg() -> np.ndarray:
    d = torch.load(
        "/workspace/data/signal/classification/SleepEEG/train.pt",
        weights_only=False, map_location="cpu",
    )
    x = d["samples"].numpy().astype(np.float32).squeeze(1)
    return ((x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-8))


def sample_windows(arr: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    n_src, L = arr.shape
    if L >= T:
        sample_ids = rng.integers(0, n_src, size=n)
        starts = rng.integers(0, L - T + 1, size=n)
        wins = np.stack([arr[s, t : t + T] for s, t in zip(sample_ids, starts)])
    else:
        sample_ids = rng.integers(0, n_src, size=n)
        wins = np.stack(
            [np.pad(arr[s], (0, max(0, T - L)))[:T] for s in sample_ids]
        )
    m = wins.mean(axis=1, keepdims=True)
    s = wins.std(axis=1, keepdims=True) + 1e-8
    return ((wins - m) / s).astype(np.float32)


def _abs_loc(x: np.ndarray, k: int, mode: str, rng: np.random.Generator) -> int:
    abs_x = np.abs(x)
    sums = np.convolve(abs_x, np.ones(k), mode="valid")
    if mode == "high":
        return int(np.argmax(sums + 1e-6 * rng.standard_normal(len(sums))))
    return int(np.argmin(sums + 1e-6 * rng.standard_normal(len(sums))))


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


# ── parametric distortion families with a tunable "strength" knob ──────
# Each function returns a distorted signal given (x, strength_scale)
# where strength_scale multiplies the family's natural distortion magnitude.

def _flip_window(x: np.ndarray, start: int, L: int, scale: float) -> np.ndarray:
    """Sign-flip within window weighted by scale (0=no flip, 1=full flip)."""
    out = x.copy()
    out[start : start + L] = (1.0 - 2.0 * scale) * out[start : start + L]
    return out


def _shift_window(x: np.ndarray, start: int, L: int, k: int, scale: float) -> np.ndarray:
    out = x.copy()
    if k == 0 or scale <= 0:
        return out
    blend = np.clip(scale, 0.0, 1.0)
    shifted = np.roll(out[start : start + L], k)
    out[start : start + L] = (1 - blend) * out[start : start + L] + blend * shifted
    return out


def _noise_window(x: np.ndarray, start: int, L: int, sigma: float,
                  noise: np.ndarray, scale: float) -> np.ndarray:
    out = x.copy()
    out[start : start + L] = out[start : start + L] + scale * sigma * noise
    return out


def _dc_offset(x: np.ndarray, c: float, scale: float, sign: int) -> np.ndarray:
    return x + sign * c * scale


def _ramp(x: np.ndarray, slope: float, scale: float, sign: int) -> np.ndarray:
    return x + sign * scale * np.linspace(0, slope, T, dtype=np.float32)


def _dropout_interp(x: np.ndarray, drop_idx: np.ndarray, scale: float) -> np.ndarray:
    """Continuous blend between identity (scale=0) and full-drop-with-linear-interp
    (scale=1). Allows MSE-matching binary search to hit any target between 0 and
    the full-drop MSE. For scale > 1, distortion saturates at the fully-dropped
    pattern (no extra severity).
    """
    a = x.copy().astype(np.float32)
    if scale <= 0 or len(drop_idx) == 0:
        return a
    s = float(np.clip(scale, 0.0, 1.0))
    # Build fully-dropped+interpolated reference once
    target = x.copy().astype(np.float32)
    target[drop_idx] = np.nan
    valid_mask = ~np.isnan(target)
    if not valid_mask.any():
        return a
    valid_pos = np.where(valid_mask)[0]
    target[drop_idx] = np.interp(drop_idx, valid_pos, target[valid_pos])
    # Continuous blend
    out = (1.0 - s) * a + s * target
    return out.astype(np.float32)


def _scale_window(x: np.ndarray, start: int, L: int, k: float, scale: float) -> np.ndarray:
    """Multiply window by 1 + scale*(k-1). scale=1 → factor k, scale=0 → identity."""
    out = x.copy()
    factor = 1.0 + scale * (k - 1.0)
    out[start : start + L] = factor * out[start : start + L]
    return out


def _flip_low_amp(x: np.ndarray, low_idx: np.ndarray, scale: float) -> np.ndarray:
    """Continuous flip across low-amp positions. scale=0 → identity, scale=1 →
    full flip, scale ∈ (0,1) → partial (toward zero then toward inverted)."""
    out = x.copy().astype(np.float32)
    if len(low_idx) == 0:
        return out
    s = float(np.clip(scale, 0.0, 4.0))
    # Allow scale > 1 so binary search hi-doubling can over-shoot when needed.
    out[low_idx] = (1.0 - 2.0 * s) * out[low_idx]
    return out


# ── Family generators: return (A_builder, B_builder, label) ────────────
def build_family(fam: str, x: np.ndarray, severity, rng: np.random.Generator):
    """Return (A_builder, B_builder, label).

    Both A and B are strength-tunable. match_mse() picks a common target
    achievable by both and binary-searches each strength independently.
    label is the structural ground truth (-1 = B more faithful, +1 = A, 0 = null).
    """
    if fam == "a":
        L = int({1: 8, 2: 16, 3: 32}[severity])
        high = _abs_loc(x, L, "high", rng); low = _abs_loc(x, L, "low", rng)
        return (
            (lambda s, _x=x, _h=high, _L=L: _flip_window(_x, _h, _L, s)),
            (lambda s, _x=x, _l=low, _L=L: _flip_window(_x, _l, _L, s)),
            -1,
        )
    if fam == "b":
        L = 16; k = int(severity)
        high = _abs_loc(x, L, "high", rng); low = _abs_loc(x, L, "low", rng)
        return (
            (lambda s, _x=x, _h=high, _L=L, _k=k: _shift_window(_x, _h, _L, _k, s)),
            (lambda s, _x=x, _l=low, _L=L, _k=k: _shift_window(_x, _l, _L, _k, s)),
            -1,
        )
    if fam == "c":
        return (
            (lambda s, _x=x, _c=severity: _dc_offset(_x, _c, s, +1)),
            (lambda s, _x=x, _c=severity: _dc_offset(_x, _c, s, -1)),
            0,
        )
    if fam == "d":
        L = 16; sigma = severity
        high = _abs_loc(x, L, "high", rng); low = _abs_loc(x, L, "low", rng)
        noise_A = rng.standard_normal(L).astype(np.float32)
        noise_B = rng.standard_normal(L).astype(np.float32)
        return (
            (lambda s, _x=x, _h=high, _L=L, _sg=sigma, _n=noise_A: _noise_window(_x, _h, _L, _sg, _n, s)),
            (lambda s, _x=x, _l=low, _L=L, _sg=sigma, _n=noise_B: _noise_window(_x, _l, _L, _sg, _n, s)),
            -1,
        )
    if fam == "e":
        return (
            (lambda s, _x=x, _sl=severity: _ramp(_x, _sl, s, +1)),
            (lambda s, _x=x, _sl=severity: _ramp(_x, _sl, s, -1)),
            0,
        )
    if fam == "f":
        abs_x = np.abs(x)
        order = np.argsort(abs_x)
        p = severity
        n_target = max(1, int(p * T))
        high_idx = order[-n_target:].copy(); low_idx = order[:n_target].copy()
        return (
            (lambda s, _x=x, _h=high_idx: _dropout_interp(_x, _h, s)),
            (lambda s, _x=x, _l=low_idx: _dropout_interp(_x, _l, s)),
            -1,
        )
    if fam == "g":
        L = 16; k = severity
        high = _abs_loc(x, L, "high", rng); low = _abs_loc(x, L, "low", rng)
        return (
            (lambda s, _x=x, _h=high, _L=L, _k=k: _scale_window(_x, _h, _L, _k, s)),
            (lambda s, _x=x, _l=low, _L=L, _k=k: _scale_window(_x, _l, _L, _k, s)),
            -1,
        )
    if fam == "h":
        abs_x = np.abs(x); thresh = severity * abs_x.max()
        low_mask = abs_x < thresh
        idx_low = np.where(low_mask)[0]
        if len(idx_low) < 2:
            return (lambda s, _x=x: _x.copy()), (lambda s, _x=x: _x.copy()), 0
        rng.shuffle(idx_low)
        half = len(idx_low) // 2
        a_idx = idx_low[:half]; b_idx = idx_low[half:]
        return (
            (lambda s, _x=x, _ai=a_idx: _flip_low_amp(_x, _ai, s)),
            (lambda s, _x=x, _bi=b_idx: _flip_low_amp(_x, _bi, s)),
            0,
        )
    if fam == "i":
        a_natural = severity
        b_natural = 2.0 - severity if severity < 1.5 else 1.0 / severity
        return (
            (lambda s, _x=x, _an=a_natural: (_x * (1 + s * (_an - 1))).astype(np.float32)),
            (lambda s, _x=x, _bn=b_natural: (_x * (1 + s * (_bn - 1))).astype(np.float32)),
            0,
        )
    if fam == "j":
        c = severity
        return (
            (lambda s, _x=x, _c=c: (_x + s * _c).astype(np.float32)),
            (lambda s, _x=x, _c=c: (_x - s * _c).astype(np.float32)),
            "_fft",
        )
    raise ValueError(fam)


# ── Mutual MSE matching: pick a common target reachable by both A and B ─
def _strength_for_target(builder, x, target, tol):
    """Binary-search strength s ∈ [0, hi] so |mse(builder(s),x) − target|/target ≤ tol.

    Returns (signal, rel_diff). hi auto-doubles up to 20× until mse exceeds target,
    capped at strength 2^20.
    """
    if target <= 0:
        sig0 = builder(0.0)
        return sig0, abs(mse(sig0, x) - target) / max(target, 1e-8)

    hi = 1.0
    mse_hi = mse(builder(hi), x)
    for _ in range(20):
        if mse_hi >= target:
            break
        hi *= 2.0
        mse_hi = mse(builder(hi), x)

    lo = 0.0
    best = None; best_diff = float("inf")
    for _ in range(MAX_BSEARCH_ITERS):
        mid = 0.5 * (lo + hi)
        sig = builder(mid)
        m = mse(sig, x)
        d = abs(m - target) / max(target, 1e-8)
        if d < best_diff:
            best_diff = d; best = sig
        if d <= tol:
            return sig, d
        if m < target:
            lo = mid
        else:
            hi = mid
    return best, best_diff


def match_mse(x, A_builder, B_builder, label_hint, tol=MSE_MATCH_REL_TOL):
    """Return (A, B, label, rel_diff). Picks a target MSE reachable by both
    families' builders (min of each's max-strength MSE) and binary-searches
    both A's and B's strength to hit the target.
    """
    # Compute each builder's max-strength MSE at hi=1.0 (natural strength).
    mse_A1 = mse(A_builder(1.0), x)
    mse_B1 = mse(B_builder(1.0), x)
    # Allow the smaller side to determine the target; the other side scales
    # down to match. This preserves the structural meaning of each side.
    target = min(mse_A1, mse_B1) if min(mse_A1, mse_B1) > 1e-8 else max(mse_A1, mse_B1)
    if target <= 1e-8:
        # both builders produce ~identity; return identity pair
        A0 = A_builder(0.0); B0 = B_builder(0.0)
        return _finalize(x, A0, B0, label_hint, 0.0)

    A_sig, dA = _strength_for_target(A_builder, x, target, tol)
    B_sig, dB = _strength_for_target(B_builder, x, target, tol)
    mse_A = mse(A_sig, x); mse_B = mse(B_sig, x)
    rel_diff = abs(mse_A - mse_B) / max(mse_A, mse_B, 1e-8)
    return _finalize(x, A_sig, B_sig, label_hint, rel_diff)


def _finalize(x, A, B, label_hint, rel_diff):
    if label_hint == "_fft":
        fx = np.abs(np.fft.rfft(x))
        fa = np.abs(np.fft.rfft(A)); fb = np.abs(np.fft.rfft(B))
        sa = float(np.dot(fx, fa) / (np.linalg.norm(fx) * np.linalg.norm(fa) + 1e-8))
        sb = float(np.dot(fx, fb) / (np.linalg.norm(fx) * np.linalg.norm(fb) + 1e-8))
        label = 1 if sa > sb else -1
    else:
        label = int(label_hint)
    return A, B, label, float(rel_diff)


def main() -> int:
    print(f"[V1 gen v2] Loading ECG…", flush=True)
    ecg = load_ecg()
    print(f"[V1 gen v2] ECG shape: {ecg.shape}", flush=True)
    print(f"[V1 gen v2] Loading SleepEEG…", flush=True)
    eeg = load_sleepeeg()
    print(f"[V1 gen v2] SleepEEG shape: {eeg.shape}", flush=True)

    n_ecg = N_PER_FAMILY // 2
    n_eeg = N_PER_FAMILY - n_ecg
    src_rng = np.random.default_rng(SEED_WIN)
    wins_ecg = sample_windows(ecg, n_ecg * len(FAMILIES), src_rng)
    wins_eeg = sample_windows(eeg, n_eeg * len(FAMILIES), src_rng)

    pair_rng = np.random.default_rng(SEED_PAIR)

    F = len(FAMILIES)
    x_arr = np.zeros((F, N_PER_FAMILY, T), dtype=np.float32)
    A_arr = np.zeros((F, N_PER_FAMILY, T), dtype=np.float32)
    B_arr = np.zeros((F, N_PER_FAMILY, T), dtype=np.float32)
    labels = np.zeros((F, N_PER_FAMILY), dtype=np.int8)
    family_id = np.zeros((F, N_PER_FAMILY), dtype=np.int8)
    source_ds = np.zeros((F, N_PER_FAMILY), dtype=np.int8)
    severity_arr = np.zeros((F, N_PER_FAMILY), dtype=np.float32)
    rel_diff_arr = np.zeros((F, N_PER_FAMILY), dtype=np.float32)

    for fi, fam in enumerate(FAMILIES):
        print(f"[V1 gen v2] family {fam} ({fi+1}/{F}) …", flush=True)
        sev_grid = SEVERITY[fam]
        for pi in range(N_PER_FAMILY):
            if pi < n_ecg:
                x_src = wins_ecg[fi * n_ecg + pi]; source_ds[fi, pi] = 0
            else:
                x_src = wins_eeg[fi * n_eeg + (pi - n_ecg)]; source_ds[fi, pi] = 1
            sev = float(sev_grid[pi % len(sev_grid)])
            A_builder, B_builder, label_hint = build_family(fam, x_src, sev, pair_rng)
            A, B, label, rel_diff = match_mse(x_src, A_builder, B_builder, label_hint)
            x_arr[fi, pi] = x_src
            A_arr[fi, pi] = A
            B_arr[fi, pi] = B
            labels[fi, pi] = label
            family_id[fi, pi] = fi
            severity_arr[fi, pi] = sev
            rel_diff_arr[fi, pi] = rel_diff

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        x=x_arr, A=A_arr, B=B_arr,
        labels=labels,
        family_id=family_id,
        source_dataset_id=source_ds,
        severity=severity_arr,
        matched_rel_mse_diff=rel_diff_arr,
        families=np.array(FAMILIES),
    )

    print(f"\n[V1 gen v2] MSE matching quality per family (target tol = {MSE_MATCH_REL_TOL:.1%}):")
    overall_ok = 0
    for fi, fam in enumerate(FAMILIES):
        rd = rel_diff_arr[fi]
        within = (rd <= MSE_MATCH_REL_TOL).mean()
        overall_ok += within
        print(
            f"  {fam}: median={np.median(rd):.3%}  p95={np.quantile(rd,0.95):.3%}  "
            f"within_tol={within:.1%}"
        )
    print(f"\n  overall within-tol rate: {overall_ok/F:.1%}")
    print(f"\n[V1 gen v2] Output: {OUT_PATH}")
    print(f"[V1 gen v2] Total pairs: {F * N_PER_FAMILY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
