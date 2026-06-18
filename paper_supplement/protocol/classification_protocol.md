---
name: classification_protocol
description: "Pre-registered protocol for the AAAI27 Classification Confirmatory section (Plan B++)."
version: v1.0
frozen_at: 2026-06-17
freeze_tag: sdsc-canonical-v5
canonical_commit: 76f30644fa06f76bf2cc55ae0b7921a03fc3fd48
authors: dlwpdud@gmail.com
ralplan_iteration: 2 (Architect STRENGTHEN + Critic ITERATE → final B++)
---

# Classification Confirmatory Protocol (AAAI27 Sec 7)

This protocol is **pre-registered at iteration-2 ralplan approval**. Any classification
analysis script (`paper_supplement/scripts/analyze_classification.py`) MUST verify
PREREG_HASH gate before computing results.

## Purpose

Test whether the forecasting paper's core claims — **C2 loss-neutrality** and
**C4 ZCR catastrophic** — generalize to time-series classification. This is the
**Classification Confirmatory** evidence; it is NOT a separate paper.

Directly addresses prior reviewer pushback:
- ICML bwZj W2: "classification benchmark 2개뿐 (Table 21 degenerate identical
  scores)" — solved by 3 in-domain datasets + 3 cross-domain transfer pairs.
- ICML 4Zmu / bwZj: "in-domain frozen-encoder classification에서만 분명한 우위"
  — solved by cross-domain transfer with explicit non-degeneracy check.
- ICLR QVmd: "cross-domain 모순 — SDSC < MSE on cross-domain" — addressed
  honestly: cross-domain pre-registered as the harder case; equivalence-margin
  loosened to ±3% accuracy.

## Scope (54 runs total, ~3 days compute on single RTX 6000 Ada)

### In-domain (27 runs)
| Dataset | Channels | T | classes | Used as |
|---|---|---|---|---|
| Epilepsy | 1 | 178 | 2 | In-domain target |
| Gesture | 3 | 206 | 8 | In-domain target |
| HAR | 9 | 128 | 6 | In-domain target |

Cells: 3 datasets × 3 losses {MSE, SDSC, ZCR} × 1 backbone (TFC pre-existing) × 3 seeds {42, 123, 2024} = **27 runs**.

### Cross-domain transfer (27 runs)
3 source-domain → target-domain pairs:

| # | Pretrain (source) | Finetune (target) | Rationale |
|---|---|---|---|
| 1 | SleepEEG (1ch, T=178) | Epilepsy (1ch, T=178) | Standard TFC pair, comparable signal |
| 2 | SleepEEG (1ch, T=178) | Gesture (3ch, T=206) | Cross-modality test |
| 3 | ECG (1ch, T=1500) | Epilepsy (1ch, T=178) | Diverse source domain (cardiac vs EEG) |

Cells: 3 pairs × 3 losses × 1 backbone × 3 seeds = **27 runs**.

**Total: 54 runs**. Seeds shared with AC-6 anchor seeds {42, 123, 2024} for cross-track comparability.

## Acceptance Criteria (locked from ralplan Critic iteration 2)

### AC-CL-1 — Pre-registration freeze
- `paper_supplement/protocol/classification_seedcells.json` frozen at Week 3 commit-hash gate.
- 54 cell IDs locked. No reordering, no scope reduction post-freeze except via new tag (sdsc-canonical-v6).

### AC-CL-2 — ZCR catastrophic threshold (pre-registered)
**Hypothesis**: ZCR loss accuracy drop > 5% vs MSE on ≥ 2 of 3 in-domain datasets.
- If ≥ 2/3 datasets satisfy → C4 (ZCR catastrophic) **confirmed cross-task**.
- If < 2/3 → report honestly: "ZCR's classification degradation is dataset-dependent."

### AC-CL-3 — Loss-neutrality threshold (pre-registered)
**Hypothesis**: |accuracy(SDSC) − accuracy(MSE)| ≤ 1% on all 3 in-domain datasets.
- If all 3 satisfy → C2 (loss-neutrality) **confirmed cross-task**.
- If any fails by > 3% → C2 framing **revised to "loss-neutrality holds in forecasting but tightens in classification"** (honest disclosure, no claim retraction).

### AC-CL-4 — Cross-domain TOST equivalence
- Paired permutation test SDSC vs MSE on cross-domain accuracy.
- TOST margin: **±3% accuracy** (looser than forecasting ±0.5% MSE due to discrete labels).
- BH-FDR at q=0.05 across 3 transfer pairs.
- **Equivalence pass** = TOST rejects H0 (not equivalent) at FDR.

### AC-CL-5 — Page-budget allocation
- Section 7 Classification: **1.5 pages text** (in AAAI main body).
- Appendix B Classification details: **0.5 pages** (full per-cell tables).
- Budget freed by moving 200-cell main table → Appendix A (1p) + Discussion trim 0.5p → 0.5p.
- Measured during Week 6 writing pass.

### AC-CL-6 — SleepEEG-subset tripwire (memory/time)
- Standard cross-domain pretrain: SleepEEG full train set (371,055 samples, ~439MB).
- If pretrain time per epoch > 30 min OR GPU memory > 40GB:
  → Switch to **SleepEEG-subset (first 50,000 samples deterministic, seed 42)**.
  → Document the subset in the cell metadata.
- Tripwire is **NOT** "fall back to in-domain only" (this was the Critic's R4 rejection).

### AC-CL-7 — Source-domain diversity (3 pairs cover distinct sources)
- Source domains: {SleepEEG (EEG), SleepEEG (EEG), ECG (cardiac)}.
- 3rd pair ECG→Epilepsy ensures **2 source modalities** (EEG and cardiac), defeating the "single source" critique on cross-domain.

## Pre-registered hypothesis table

| Claim | Prediction | Threshold (AC) | Verification source |
|---|---|---|---|
| C4 (ZCR catastrophic) holds cross-task | ZCR acc drop > 5% on ≥ 2/3 in-domain | AC-CL-2 | analyze_classification.py |
| C2 (loss-neutrality) holds cross-task | acc(SDSC) − acc(MSE) ≤ 1% on all 3 in-domain | AC-CL-3 | analyze_classification.py |
| Cross-domain non-degenerate | TOST passes at ±3% acc margin (BH q=0.05) | AC-CL-4 | analyze_classification.py |
| Cross-domain SDSC < MSE (ICLR QVmd) | EXPECTED: SDSC slightly worse on cross-domain target | Reported honestly | analyze_classification.py |

The ICLR QVmd objection is treated as **expected** and explained in Sec 6 Discussion via the amplitude-tolerance argument from the forecasting paper.

## Scheduling

| Week | Activity |
|---|---|
| **Week 3 (06-22~)** | Infrastructure: SDSC port to TFC verified; ZCR added to loss.py + main.py; config files (Gesture, HAR, ECG) committed; seedcells.json frozen. |
| **Week 4 (06-29~)** | Launch 54-run sweep (~3 days). Hand-off brief Sec 15 placeholder added. |
| **Week 5 (07-06~)** | Sweep complete → run analyze_classification.py → fill Section 7 outline. |
| **Week 6 (07-13~)** | Section 7 final draft in hand-off brief. AC-CL-5 page measurement. |
| Week 7-10 | Forecasting paper writing in hand-off mode. Classification merged. |

## Falsification gates

This protocol is **falsifiable**. The following outcomes require honest reporting, NOT scope retraction:

| Observation | Required reporting |
|---|---|
| ZCR shows > 5% degradation on only 1/3 in-domain | "ZCR's classification degradation is dataset-dependent" |
| acc(SDSC) − acc(MSE) > 3% on any in-domain | "Loss-neutrality tightens in classification; SDSC/MSE not fully interchangeable" |
| Cross-domain TOST fails on > 1/3 pairs at ±3% | "Cross-domain transfer remains a harder case; SDSC ≈ MSE only within ±X% (observed)" |
| SleepEEG-subset tripwire triggered | Documented in AC-CL-6 with subset commit-hash |

## Pre-registration commitment

After this protocol freezes (Week 3 commit), the following are **immutable**:
- AC thresholds (5%, 1%, ±3%)
- 54 cell IDs in seedcells.json
- 3 seeds {42, 123, 2024}
- TOST margins (cross-domain ±3% acc)
- Page allocation (1.5p + 0.5p)

Any change requires new sdsc-canonical-vN tag and disclosure of pre/post-freeze diff.

## References

- Plan B++ (this ralplan iteration): `paper_supplement/AAAI27_metric_validation_plan.md` (forecasting)
- Forecasting V1 evidence: `paper_supplement/protocol/V1_results_summary.md`
- Forecasting AC-6 anchor: `paper_supplement/protocol/AC6_results_summary.md`
- Hand-off brief: `paper_supplement/SDSC_AAAI27_paper_writing_brief.md` (Section 15 to be added)
- Frozen seedcells: `paper_supplement/protocol/classification_seedcells.json`
