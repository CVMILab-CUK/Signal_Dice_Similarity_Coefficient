# SDSC AAAI27 — Metric Validation Plan (v5, pending approval)

**Status**: pending approval | **Authored**: 2026-06-01 | **Consensus**: ralplan iter 5/5 APPROVE
**Target venue**: AAAI 2027 (deadline 2026-08-15) | **Fork venue**: TMLR (rolling)
**Iterations**: Planner → Arch STRENGTHEN → Critic ITERATE → Arch STRENGTHEN → Critic ITERATE → Arch STRENGTHEN → Critic APPROVE

---

## Architecture Decision Record (ADR)

### Decision
Pivot from **loss-claim** ("SDSC as loss outperforms MSE") to **metric-first complementary** ("SDSC measures sign-and-magnitude structure where MSE is blind; complementary to MSE with characterized DC-offset failure boundary"). Land V1 (MSE-equivalent pair metric validation) + V3 (reconstruction-tensor disagreement) inside 10 weeks (2026-06-08 to 2026-08-15) with a 3-branch fork at week 7 (2026-07-22).

### Drivers
1. **Empirical (decisive)**: 200-cell grid (4 backbones × 7 datasets × 8 losses, N=1) shows loss-neutrality. SDSC wins 0 of 7 datasets on downstream MSE. C2 claim ("SDSC as loss") is empirically dead.
2. **Reviewer-history (cumulative)**: 3 prior rejections (AAAI26 / ICLR26 / ICML26) flagged the same axes — marginal gain, single backbone, missing 1-bit-sign / discretization baselines, single-seed statistical fragility.
3. **Time-to-evidence (binding)**: 10 weeks. Grid extension (TimeMixer / Mamba / N=3 on full 200) costs ≥3 weeks and produces *more* loss-neutrality evidence, not metric evidence. Metric validation (V1+V3) is the only lever that changes review trajectory.
4. **Reject-probability honest estimate**: current plan (grid + narrative only) = 70-80% reject. With V1+V3 landing = 30-40% reject. Single biggest delta.

### Alternatives considered
- **Option A**: V1 only — rejected (no ZCR baseline → re-triggers ICML 4Zmu attack verbatim; no real-checkpoint sanity)
- **Option B**: V2 model selection only — rejected (downstream-as-judge framing collapses metric-first thesis; task-definition subjective)
- **Option C**: V3 only — rejected (cannot carry headline; weak alone)
- **Option D**: V1 + V3 hybrid — **CHOSEN**, strengthened through 4 architect rounds + 2 critic rounds into plan v5
- **Option E**: Early fork to TMLR (loss-neutrality + qualitative viz) — **retained as week-7 contingency** (AC-7 branch 3)

### Why chosen (Option D + plan v5 synthesis)
- V1 anchors the *positive* metric claim (SDSC distinguishes structurally meaningful pairs MSE cannot)
- V3 anchors the *complementarity* claim (real reconstructions show disagreement correlated with structural fidelity)
- 60-run statistical anchor (12 unique configs × N=5) defends loss-neutrality at TOST ±1.4% MDE without the infeasible 200-cell × N=5 cost (~49 days infeasible → 2 days feasible)
- 10-family preregistration with bracketed predictions (ceilings g, floors h+j, neutral i, wins a-f) defangs both circular-construction (ICLR tXhx) and ZCR-equivalence (ICML 4Zmu) attacks simultaneously
- 3-branch fork at week 7 (AAAI-clean / AAAI-honest / TMLR) preserves 10-week budget under any V1 outcome

### Consequences
- **Positive**: Plan converts strategic disadvantage (loss claim dead) into honest narrative asset (loss-neutrality is the *evidence* for needing a metric). Compute footprint 208 GPU-h vs 336 available — 38% buffer.
- **Negative**: Paper makes 2 preregistered SDSC failure claims (families h and j). Honesty trade — converts to strength under "complementary" framing (Δ13), risks negative-result quoting under any other framing.
- **Constraints**: Δ13 framing must be enforced via final-pass manuscript lint (forbidden tokens: dominates/outperforms/superior/beats/better-than-MSE). 60-run anchor table must appear in main body BEFORE 200-cell descriptive grid (per Critic M2). 5-seed pilot on (MSE, SimMTM, ETTh1) before committing to N=5 main run (per Critic M3).

### Follow-ups
- Week 0 (2026-06-01~07): execute AC-5 SDSC canonical lock + AC-10 data inventory + micro-benchmark
- Week 7 (2026-07-22): AC-7 fork decision evaluation
- Post-submission: dedicated "Coverage and Residual Risk" subsection (Critic M4)

---

## Plan v5 — Acceptance Criteria

### AC-1 — V1 pre-registration (10 families, bracketed predictions)
**Files**: `paper_supplement/protocol/V1_protocol.md` + `paper_supplement/protocol/PREREG_HASH`
**Pre-registered families** (a–j, N≥500 pairs each):
| Family | Distortion | SDSC prediction | ZCR/1-bit prediction |
|---|---|---|---|
| a | sign-inversion (random windows) | WIN | WIN |
| b | phase shift amplitude-preserving | WIN | WIN |
| c | DC offset (small) | WIN | WIN |
| d | HF noise burst | WIN | WIN |
| e | trend addition (linear ramp) | WIN | WIN |
| f | sample dropout + interpolation | WIN | WIN |
| g | sign-preserving structural damage (adversarial) | **PASS** | FAIL |
| h | low-amplitude meaningless sign flips (adversarial) | **FAIL (ρ<0.3)** | FAIL |
| i | scale-only (k∈{0.5,1.5,2.0}) (null) | **≈ MSE** | ≈ MSE |
| j | DC offset injection (adversarial — Critic M2 fix) | **LOSE to MSE** | LOSE |

**Acceptance**: protocol file committed before week 2 with git tag `sdsc-canonical-v1`, V1 analysis notebook reads commit hash from `paper_supplement/protocol/PREREG_HASH` frontmatter and aborts if uncommitted.

### AC-2 — V1 baselines (ZCR + 1-bit + 2-bit μ-law)
Co-baselines: {MSE, MAE, **ZCR (differentiable, soft-sign relaxation from speech literature)**, **1-bit MSE = MSE(sign(z(E)), sign(z(R)))** z=channel-wise z-score, **2-bit μ-law MSE**, PCC, SI-SNR, SDSC}.
Per-family ρ + bootstrap CI (N≥1000 resamples) + paired permutation SDSC vs ZCR + SDSC vs 1-bit-MSE.
Pre-registered ordering on family (g): SDSC > 2-bit > 1-bit > ZCR.

### AC-3 — 3-tier ground truth ladder + dual-test
- **Tier A (headline)**: parametric synthetic distortion on real signals from `/workspace/data/signal/classification/{ECG, Epilepsy, EMG}` + ETT/Weather. Pass = (SDSC ρ > ZCR ρ via paired permutation p<0.05) AND (bootstrap 95% CI of ρ_SDSC − ρ_ZCR excludes 0).
- **Tier B (robustness)**: pretrained classifier on PhysioNet labeled subset. Documented as "downstream-dependent, inherits loss-neutrality caveats."
- **Tier C (sanity, optional)**: N=50-100 SQI rubric scoring (Clifford et al ECG quality index).

### AC-4 — V3 reframe + budget inversion
- **Week 0 day 1-2**: recoverability audit + 50 min/cell micro-benchmark.
- Reconstruction tensors NOT saved in pretrain ckpts → V3 = pretrain re-pass with reconstruction logging.
- **Primary scope**: {SimMTM, DLinear} × {ETTh1, Weather} × 8 losses × 3 seeds = **48 cells** (~120 GPU-h, 5 days)
- **Stretch goal**: 192 cells (4 backbones), contingent on week-0 micro-benchmark hitting ≤30 min/cell mean
- **Tripwire**: if day-7 mean wallclock > 30 min/cell → lock 48-cell primary, stretch 192 locked out

### AC-5 — Canonical SDSC lock (corrected codebase facts)
Lock the **two real divergences** between `libs/metric.py` and `SimMTM_Forecasting/utils/metrics.py`:
- (i) **H(0) convention**: lock H(0)=0 (`utils/metrics.py:33`). `libs/metric.py:25` uses H(0)=1 — deprecate.
- (ii) **Reduction axis**: lock per-sequence `sum(dim=-1)` then mean (`libs/metric.py:30` style). `utils/metrics.py:38` does global sum — deprecate.
- (iii) **Hard/soft consolidation**: single class with `alpha` switch (utils-style).
- (iv) **α-aware tie tests** (Critic M1 fix):
  - Analytical tie (E=R=0): tol = 1e-7
  - Numerical near-tie at δ=1e-3: tol(α=1)=2.5e-4, tol(α=10)=2.5e-3, tol(α=100)=2.5e-2
- (v) Git tag `sdsc-canonical-v1` on lock commit. Test file: `tests/test_sdsc_canonical.py`.

### AC-6 — Loss-neutrality statistical anchor (60-run inferential subset)
- **Scope (Critic C1 fix)**: {MSE, SDSC, ZCR} × {ETTh1, Weather} × {SimMTM, DLinear} = 12 unique configs × N=5 seeds = **60 cell-runs**
- **Frozen in `paper_supplement/protocol/ac6_seedcells.json`** at week-1 commit-hash gate
- **Equivalence test**: TOST with ±2% MSE margin (MDE at N=5 ≈ 1.4% MSE)
- **Multiple comparison**: Benjamini-Hochberg FDR at q=0.05
- **Seeds**: `[42, 123, 2024, 7, 1729]` primary; `[2025, 360, 911]` escalation
- **Auto-escalate**: per-cell independent; variance = (std/mean)×100 of test MSE; threshold 0.7%
- **Hard tripwire**: if >25% of 60 cells escalate to N=8 → downgrade to N=5 + post-hoc **Bayesian shrinkage** (informative priors, NumPyro). Documented in Appendix C.
- **Stop-condition cascade**: if N=8 also exceeds variance → mark cell as "unstable, excluded from TOST" + report in Limitations
- **Pilot (Critic M3 fix)**: 5-seed pilot on (MSE, SimMTM, ETTh1) BEFORE committing to 60-run scope; if observed std > 0.8% MSE → escalate to N=7 OR widen TOST to ±1.6%

### AC-7 — Three-branch fork at week 7 (2026-07-22)
- (A) SDSC ρ > {MSE, MAE, PCC, SI-SNR} on **≥6/8 wins-expected families (a-g)** AND
- (B) SDSC ρ > ZCR on family g (paired permutation p<0.05 + bootstrap CI excludes 0) AND
- (C) SDSC ρ ≥ 1-bit-MSE on family g (CI tie acceptable)
- **All 3 pass → AAAI-clean path** (full metric-first claim)
- **(A) only (B or C fail) → AAAI-honest path** (explicit ZCR/1-bit positioning, Limitations expanded)
- **(A) fails → TMLR pivot** (loss-neutrality + V1 honest result as standalone metric paper, target rolling submission week 9-10)

### AC-8 — Consistency-trap defense paragraph
Discussion section ≥150 words covering (a) consistency-trap framing, (b) amplitude-tolerance argument ("downstream is amplitude-tolerant; clinical labels are amplitude-tolerant; reconstruction fidelity is not amplitude-tolerant"), (c) explicit citation of 200-cell loss-neutrality.

### AC-9 — Honest failure / coverage subsection
Dedicated subsection (~0.5 page, Critic M4 fix): "Coverage and Residual Risk" with:
- (i) Family (h) preregistered SDSC failure
- (ii) Family (j) DC offset preregistered SDSC loss (exploiting ICLR tXhx offset bias)
- (iii) Trend-dominated signal limitation (ICML 4Zmu KQ1)
- (iv) Explicit "we cannot claim coverage of failure modes outside families (a-j)" — falsifiable boundary

### AC-10 — Week-0 data inventory ✅ (confirmed)
- `/workspace/data/signal/classification/{ECG, EMG, Epilepsy, FD-A, FD-B}` confirmed available
- Sampling rate / channel count / length documented in `paper_supplement/protocol/V1_data_inventory.md`

### AC-11 — Reproducibility + commit-hash gate
- Global seeds = `[42, 123, 2024, 7, 1729]` for AC-6, `[2023]` for existing N=1 grid
- Pre-registration enforcement: V1 analysis notebook reads `git rev-parse HEAD:paper_supplement/protocol/` and aborts if uncommitted (scoped to `protocol/` subdir, allows notebook iteration)
- All code/notebooks in `paper_supplement/` committed to main with git tags per milestone

### AC-12 — 200 vs 60 scope reconciliation (Architect iter-4 fix)
**Mandatory paper text**: "We report a 200-cell exploratory grid (N=1) demonstrating consistent loss-neutrality trends, complemented by a 60-run statistical anchor (3 losses × 2 datasets × 2 backbones × 5 seeds = 12 unique configurations) providing TOST equivalence inference at ±1.4% MDE."
- 60-run anchor table appears in **main body BEFORE** 200-cell descriptive grid (Critic M2)
- 200-cell grid table moved to Appendix with caption: "descriptive trends, N=1, not statistical inference; see Table X for inferential anchor"

### AC-13 — Complementary framing directive (Architect iter-4 fix)
**Locked positioning**: SDSC = "complementary to MSE with characterized DC-offset failure boundary"
- **Abstract/Intro/Conclusion**: NO "dominates" / "outperforms" / "superior" / "beats" / "better than MSE"
- **Replace with**: "complements" / "characterizes" / "distinguishes" / "where MSE is blind"
- **Final-pass lint** (Critic M1): scan full manuscript for forbidden tokens before submission

---

## Pre-mortem (7 scenarios)

| # | Scenario | Prob | Impact | Mitigation |
|---|---|---|---|---|
| R1 | SDSC ≈ ZCR on adversarial family g | M | V1 collapses | AC-7 branch 3 (TMLR pivot) preregistered |
| R2 | Recon tensors not recoverable, V3 re-pass | **H** (confirmed) | +2 weeks | Pre-budgeted, ETTh1+Weather only |
| R3 | Tier C SQI doesn't transfer to ETT/Weather | H | Tier C optional drops | Headline = Tier A only |
| R4 | AC-6 stats slippage (TOST/FDR/Bayesian unfamiliarity) | M | AC-7 fork uninformed | Buffer 3 days week 6 |
| R5 | EEG/ECG sampling-rate mismatch ETT/Weather (96 timesteps) | M | Tier A scope shrinks | Resample/crop utility week 1 |
| R6 | Pre-registration commit-hash discipline failure | L | Pre-reg credibility destroyed | Notebook abort enforcement (AC-11) |
| R7 | Family h unexpectedly passes SDSC | L | Falsifiability collapse | Pre-registered ρ < 0.3 threshold on (h) |

---

## GPU-hour ledger (208 GPU-h vs 336 available = 38% buffer)

| 작업 | cells × seeds × min/cell | GPU-h | days |
|---|---|---|---|
| AC-4 V3 primary | 48 × 3 × 50 | 120 | 5 |
| AC-6 anchor | 60 runs × 50 | 50 | 2 |
| AC-6 escalation buffer (≤25%) | 45 runs × 50 | 38 | 1.5 |
| **Total** | | **208 GPU-h** | **~9 days** |

---

## Timeline (10 weeks)

| Week | V1 | V3 (48 cells) | AC-6 (60 cells) | Writing |
|---|---|---|---|---|
| 0 (06-01~07) | AC-5 lock + AC-10 inventory | recoverability + 50 min/cell verify | seedcells.json draft | — |
| 1 (06-08) | AC-11 + S1 protocol + S2 baselines + pilot M3 | re-pass start | AC-6 cells frozen + start | loss-neutrality prelim |
| 2 (06-15) | Tier A run (10 families) | continue | continue + variance check | TOST+FDR draft |
| 3 (06-22) | Tier B run | analysis | escalation if needed | intro draft |
| 4 (06-29) | Tier C optional | writeup | TOST + Bayesian if escalated | V1 writeup |
| 5 (07-06) | analysis + paired permutation | — | — | Section 4/5 draft |
| 6 (07-13) | bootstrap CI per family | — | — | **AC-6 camera-ready 07-15** |
| **7 (07-20)** | **AC-7 fork checkpoint 07-22** | — | — | TMLR branch ready |
| 8-9 | revisions | — | — | AC-8 + AC-9 + AC-13 lint |
| 10 (08-10) | polish | — | — | submit 08-15 |

---

## Critic-noted writing-stage guardrails (M1-M4, not plan-level)

1. **M1 forbidden-token lint**: scan full manuscript for `{dominates, outperforms, superior, beats, better than MSE}` before submission
2. **M2 table ordering**: 60-run anchor table in main body BEFORE 200-cell descriptive grid; latter to Appendix
3. **M3 pre-flight pilot**: 5-seed pilot on (MSE, SimMTM, ETTh1) BEFORE committing to N=5 scope; escalate to N=7 OR widen ±1.6% if std > 0.8%
4. **M4 dedicated coverage subsection**: 0.5-page "Coverage and Residual Risk" subsection (AC-9 expansion)

---

## Status: pending approval

This plan is the consensus output of ralplan iter 1-5 (3 Architect + 2 Critic rounds). The Critic's final verdict was APPROVE with 4 writing-stage MAJOR guardrails (M1-M4) carried as a pre-submission checklist, not plan revisions.

**Execution options** (user must choose):
1. Approve via team (parallel coordinated agents)
2. Approve via ralph (sequential persistent loop)
3. Approve after clearing context (fresh session)
4. Request changes (re-open consensus loop)
5. Reject (abandon plan)

No execution will begin until explicit approval is captured.
