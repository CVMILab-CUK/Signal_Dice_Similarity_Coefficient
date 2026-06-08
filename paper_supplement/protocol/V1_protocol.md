---
name: V1_protocol
description: Pre-registered protocol for the V1 metric-validation experiment (AAAI27 plan v5 AC-1).
version: v1.0
frozen_at: 2026-06-08
freeze_tag: sdsc-canonical-v1
canonical_commit: 76f30644fa06f76bf2cc55ae0b7921a03fc3fd48
authors: dlwpdud@gmail.com
---

# V1 Pre-Registration Protocol (10 families × N=500 pairs)

This protocol is **frozen at the iteration-5 approval commit**. Any V1 analysis
notebook MUST verify the PREREG_HASH gate before computing results. Edits to
this document after the freeze require a new tag (`sdsc-canonical-v2`).

## Purpose

Test the hypothesis that **SDSC, as a similarity metric, distinguishes
structurally meaningful reconstruction pairs that MSE cannot distinguish**,
on signed time-series signals. Counter-hypothesis: SDSC is reducible to
ZCR + magnitude weighting (ICML 4Zmu attack); we directly co-evaluate ZCR,
1-bit MSE, and 2-bit μ-law MSE alongside SDSC.

## Pair-construction recipe (universal)

For every family (a)–(j), we produce N=500 pairs of the form:
```
(reference signal x_i,
 distorted signal A_i = T_i^{(family)}(x_i, severity_i),
 distorted signal B_i = U_i^{(family)}(x_i, severity_i))
```
such that `MSE(A_i, x_i) ≈ MSE(B_i, x_i)` within tolerance ≤ 2% relative.
Each metric M ∈ {MSE, MAE, ZCR, 1-bit MSE, 2-bit μ-law MSE, PCC, SI-SNR, SDSC}
ranks (A_i, B_i) by `M(A_i, x_i) − M(B_i, x_i)`. Ground-truth label = +1 if
A_i is more structurally faithful per the family's pre-registered notion,
−1 otherwise.

Spearman ρ is computed between metric rankings and ground-truth ranking on
the N=500 set.

## Source signals

| Tier | Source | N samples available | Window size used | Pair count strategy |
|---|---|---|---|---|
| A primary | ECG (MIT-BIH derived, /workspace/data/signal/classification/ECG/train.pt) | 43,673 × 1500 | 96 (non-overlapping windows, 15 per sample → ~655k) | 500 pairs/family, stratified by source-sample-ID |
| A secondary | SleepEEG (/workspace/data/signal/classification/SleepEEG/train.pt) | 371,055 × 178 | 96 (left-aligned crop) | 500 pairs/family, stratified by source-sample-ID |
| A forecasting | ETTh1, Weather validation sets | varies | 96 (existing seq_len) | 500 pairs/family from val set |
| B downgraded | TF-C-style classifier oracle | SKIPPED (V1_data_inventory.md notes) | — | Robustness section cites TF-C; no separate experiment |
| C optional | SQI rubric on subset | 50–100 | 96 | Sanity tier; not headline |

Random seed for pair sampling: `42` (deterministic).
Random seed for window selection within long sequences: `123` (deterministic).
Pair-ID file (frozen at week-1 collection): `paper_supplement/protocol/V1_pair_index.npz`.

## Ten distortion families and pre-registered predictions

Each prediction is **frozen** before any V1 result is computed.

### Family (a) — sign-inversion in random windows
**Distortion**: invert sign on a randomly chosen contiguous window of length
`L ∈ {8, 16, 32}` (severity 1, 2, 3 respectively).
**A vs B construction**: A inverts a HIGH-amplitude window; B inverts a
LOW-amplitude window. MSE matches by amplitude scaling.
**Ground truth**: B is more structurally faithful (low-amplitude flip is less
perceptually disruptive). Label B = +1.
**Pre-registered prediction**: SDSC ρ > 0.6 (WIN). ZCR ρ > 0.6 (WIN; sign flip is detectable). MSE ρ ≈ 0 (cannot distinguish).

### Family (b) — phase shift amplitude-preserving
**Distortion**: cyclically shift signal by k samples, `k ∈ {1, 3, 5}`.
**A vs B construction**: A shifts at high-amplitude region; B at low-amplitude.
**Ground truth**: B more faithful (less perceptually disruptive shift). Label B = +1.
**Pre-registered prediction**: SDSC ρ > 0.5 (WIN, but weaker than (a)). PCC ρ > 0.7 (WIN). MSE ρ ≈ 0.

### Family (c) — DC offset (small)
**Distortion**: add constant `c ∈ {0.05, 0.10, 0.20}` (relative to signal range).
**A vs B construction**: A adds positive offset; B adds negative offset (same magnitude).
**Ground truth**: NULL — both are equally distorted from structural fidelity. Label = 0 (random rank).
**Pre-registered prediction**: SDSC ρ ≈ 0 (no preference). MSE ρ ≈ 0. ZCR ρ ≈ 0. All metrics indistinguishable.

### Family (d) — high-frequency noise burst
**Distortion**: add Gaussian noise `N(0, σ²)` in a contiguous window, `σ ∈ {0.1, 0.3, 0.5}`.
**A vs B construction**: A at high-amplitude window; B at low-amplitude window.
**Ground truth**: B more faithful (low-amplitude noise less disruptive). Label B = +1.
**Pre-registered prediction**: SDSC ρ > 0.5 (WIN — magnitude weighting helps). MSE ρ < 0.3 (weak; cannot localize).

### Family (e) — trend addition (linear ramp)
**Distortion**: add linear ramp from 0 to `slope ∈ {0.1, 0.3, 0.5}` over the window length.
**A vs B construction**: A adds positive slope; B adds negative slope.
**Ground truth**: NULL — both equally distorted. Label = 0 (random).
**Pre-registered prediction**: All metrics ρ ≈ 0. (Honesty check.)

### Family (f) — sample dropout + interpolation
**Distortion**: drop random samples with rate `p ∈ {0.05, 0.10, 0.20}` and linearly interpolate.
**A vs B construction**: A drops high-amplitude samples; B drops low-amplitude.
**Ground truth**: B more faithful. Label B = +1.
**Pre-registered prediction**: SDSC ρ > 0.4 (WIN — magnitude weighting). MSE ρ ≈ 0.

### Family (g) — sign-preserving structural damage **[adversarial / steelman]**
**Distortion**: scale up a high-amplitude window by `k ∈ {1.5, 2.0, 3.0}`
WITHOUT flipping any signs (structurally damaging but sign-preserving).
**A vs B construction**: A damages a structurally important window
(detected by max(|x|) localization); B damages a less important window.
**Ground truth**: B more faithful (damage at less-important window). Label B = +1.
**Pre-registered prediction**:
  - SDSC ρ > 0.6 (**PASS** — magnitude weighting captures this).
  - ZCR ρ ≈ 0 (**FAIL** — sign-only metric is blind to magnitude damage).
  - 1-bit MSE ρ ≈ 0 (FAIL).
  - 2-bit μ-law MSE ρ > 0.3 (PARTIAL — multi-bit captures some structure).
**This family is the critical evidence for SDSC ≠ ZCR.** Plan v5 AC-7 fork
criterion (B) requires SDSC ρ > ZCR ρ on family (g) with paired permutation
p<0.05 and bootstrap CI excluding 0.

### Family (h) — low-amplitude meaningless sign flips **[adversarial / honesty]**
**Distortion**: flip signs in a low-amplitude (|x| < 0.1 × max(|x|)) region.
**A vs B construction**: A flips at very low amplitude; B flips at slightly higher.
**Ground truth**: NULL — both flips are perceptually meaningless. Label = 0 (random).
**Pre-registered prediction**:
  - SDSC ρ < 0.3 (**FAIL** — preregistered honest failure).
  - This is a falsifiability check. SDSC should NOT spuriously succeed here.
**If SDSC unexpectedly succeeds**, paper reports honestly and adjusts coverage claim.

### Family (i) — scale-only distortion **[null]**
**Distortion**: multiply by `k ∈ {0.5, 1.5, 2.0}` globally.
**A vs B construction**: A scales by 2.0; B scales by 0.5. Both preserve all signs.
**Ground truth**: NULL — scale is amplitude-only, no structural information lost.
Label = 0 (random).
**Pre-registered prediction**:
  - SDSC ρ ≈ MSE ρ ≈ 0.
  - This is the **non-circular** evidence that SDSC is not "always wins."
  - ZCR ρ ≈ 0, 1-bit ρ ≈ 0 (all sign-aware metrics agree — sign unchanged).

### Family (j) — DC offset injection **[adversarial / honesty]**
**Distortion**: add constant offset `c ∈ {0.3, 0.5, 1.0}` (large enough to
shift some sign decisions near zero crossings).
**A vs B construction**: A adds offset that crosses zero crossings; B adds
the same magnitude but in opposite direction (also crossing zero crossings).
**Ground truth**: B more faithful if its crossing pattern preserves cycle
shape better (this is ground-truth labeled by computing the spectral
coherence between A/B and the reference; the higher-coherence one is label +1).
**Pre-registered prediction**:
  - SDSC ρ < MSE ρ (**LOSE TO MSE**).
  - This is the **honesty-as-strength** family. SDSC is documented to over-rate
    when DC offset dominates sign decisions (ICLR tXhx offset bias).
  - **If SDSC unexpectedly beats MSE here**, our coverage claim shrinks
    accordingly — reported honestly.

## Co-baselines for V1

For each family, every metric in this set is computed:
```
{MSE, MAE, ZCR (differentiable soft-sign), 1-bit MSE, 2-bit μ-law MSE, PCC, SI-SNR, SDSC (canonical)}
```

Reference implementations:
- MSE, MAE — `torch.nn.MSELoss`, `torch.nn.L1Loss`
- ZCR — `SimMTM_Forecasting/utils/baselines/zcr_diff.py:DiffZCRLoss`
- 1-bit MSE — `SimMTM_Forecasting/utils/baselines/quantized_mse.py:one_bit_mse`
- 2-bit μ-law MSE — `SimMTM_Forecasting/utils/baselines/quantized_mse.py:two_bit_mu_law_mse`
- PCC — `SimMTM_Forecasting/utils/metrics.py:pearson_correlation`
- SI-SNR — `SimMTM_Forecasting/utils/metrics.py:si_snr`
- SDSC — `SimMTM_Forecasting/utils/sdsc_canonical.py:SignalDiceCanonical` (alpha=None)

## Pre-registered ordering on family (g)

Pre-registered prediction (to be confirmed or refuted by V1):
```
SDSC ρ > 2-bit μ-law MSE ρ > 1-bit MSE ρ > ZCR ρ
```

Statistical test: paired permutation between SDSC ρ and each baseline ρ;
bootstrap 95% CI on (ρ_SDSC − ρ_baseline) must exclude 0 for "SDSC wins"
claim.

## Statistical analysis specification

For each family (a)–(j) and each metric:
1. Compute per-pair metric difference and rank.
2. Spearman ρ between metric ranking and ground-truth label.
3. Bootstrap 95% CI: 1000 resamples.
4. Paired permutation test SDSC vs each baseline: 10,000 permutations, two-sided.
5. Multiple-comparison correction: Benjamini-Hochberg FDR at q=0.05 across all
   pair-of-(family, baseline) comparisons.

## Falsification criteria (the gate)

V1 is reported as **successful** if and only if ALL of:
1. Family (g): SDSC ρ > ZCR ρ with paired permutation p<0.05 and bootstrap CI
   of (ρ_SDSC − ρ_ZCR) excludes 0.
2. Family (g): SDSC ρ ≥ 1-bit MSE ρ (within CI overlap acceptable).
3. Family (h): SDSC ρ < 0.3 (preregistered honest failure honored).
4. Family (i): SDSC ρ within ±0.1 of MSE ρ (null preserved).
5. At least 6 of families (a, b, d, f, g) show SDSC ρ > 0.4 (WIN families).
6. Family (j): SDSC ρ < MSE ρ (honest failure honored).

Conditions 1–6 together feed plan v5 AC-7 fork decision at Week 7 (2026-07-22).

## What this protocol commits NOT to do

To preserve pre-registration credibility, after data collection begins
(Week 1, 2026-06-08+), we will NOT:
- Change distortion family definitions or severity levels.
- Change N=500 per family.
- Change baseline metric set.
- Move thresholds in conditions 1–6.
- Add a family post hoc to "fix" a failed prediction.

Any such change requires a new `sdsc-canonical-v2` tag and full disclosure
of what was changed and why.

## Frozen artifacts (to be created at Week 1)

- `paper_supplement/protocol/V1_pair_index.npz` — pair IDs (source sample
  index, window offset, distortion parameters)
- `paper_supplement/protocol/V1_pair_index_hash.txt` — SHA-256 of the npz file
- `paper_supplement/protocol/V1_pair_samples.png` — visualization of 1 pair
  per family (sanity check, frozen with hash)

## References

- Plan v5: `paper_supplement/AAAI27_metric_validation_plan.md`
- Pre-registration gate: `paper_supplement/protocol/PREREG_HASH`
- Data inventory: `paper_supplement/protocol/V1_data_inventory.md`
- M3 pilot: `paper_supplement/protocol/M3_pilot_results.md`
- Soft-sign ZCR literature: speech processing community; standard
  `tanh(α·x)` relaxation of `sign(x)`.
