---
name: classification_protocol_v2
description: "C-4 (metric-only) + C-5 (loss-in-recon-head) cross-backbone generalization. Supersedes classification_protocol.md (v1, B++) which assumed mislabeled SimMTM-Cls was TFC."
version: v2.0
frozen_at: 2026-06-19
freeze_tag: sdsc-canonical-v6 (pending)
canonical_commit: 76f30644fa06f76bf2cc55ae0b7921a03fc3fd48
supersedes: classification_protocol.md
authors: dlwpdud@gmail.com
discovery: TFC class in SimMTM_Classification is actually SimMTM-Cls (no FFT, no time-freq); v1 protocol's "TFC backbone" label was wrong.
---

# Classification Generalization Protocol v2 (cross-backbone + cross-task)

This protocol supersedes v1 (Plan B++). v1 used SimMTM_Classification/code which
turns out to be a SimMTM variant, not original TFC. v2 adds **two real SSL
backbone families** (TF-C original, TS2Vec) and runs **two orthogonal
generalization tests** that map to distinct reject objections:

| Test | Question answered | Forecasting parallel |
|---|---|---|
| **C-4 (metric)** | Does SDSC measure representation quality across SSL frameworks? | V1 metric validation extended cross-backbone + cross-task |
| **C-5 (loss-in-head)** | Does loss choice in a post-hoc reconstruction head affect downstream when the encoder is frozen? | 200-cell loss-neutrality + V3 model selection cross-backbone + cross-task |

## Why both, not one

C-4 alone leaves "but you didn't test loss-as-loss" open.
C-5 alone leaves "but you didn't test metric-as-metric" open.
Together they cover both axes — directly addresses ICML 4Zmu, bwZj, ICLR 5F1V,
QVmd objections simultaneously.

## Backbones (3 families)

| Backbone | Source | Pretrain loss | Has recon head? | Wrapper file |
|---|---|---|---|---|
| **SimMTM-Cls** | SimMTM_Classification/code/ (already running) | Contrastive + MSE recon | Yes (native) | existing main.py |
| **TF-C (real)** | backbones/TFC/code/TFC/ (cloned upstream) | NT-Xent time + freq + cross | No (we ADD for C-5) | paper_supplement/scripts/tfc_wrapper.py |
| **TS2Vec** | backbones/TS2Vec/ (cloned upstream) | Hierarchical instance + temporal contrastive | No (we ADD for C-5) | paper_supplement/scripts/ts2vec_wrapper.py |

## Datasets (3 in-domain + 3 cross-domain transfer)

| Type | Source / Target | Channels × T | Used for |
|---|---|---|---|
| in-domain | Epilepsy (1×178) | C-4 + C-5 |
| in-domain | Gesture (3×206) | C-4 + C-5 |
| in-domain | HAR (9×128) | C-4 + C-5 |
| xdomain | SleepEEG → Epilepsy | C-4 + C-5 |
| xdomain | SleepEEG → Gesture | C-4 + C-5 |
| xdomain | ECG → Epilepsy | C-4 + C-5 |

Same as v1 (proven dataset spec); the addition is the backbone axis.

## C-4 design (metric-only generalization)

For each (backbone, dataset):
1. Pretrain encoder with backbone's **original** loss (NT-Xent for TF-C, hierarchical contrastive for TS2Vec, contrastive+MSE for SimMTM-Cls).
2. Freeze encoder.
3. Compute representation z = encoder(x) on test set.
4. Reconstruct x̂ from z via a tiny linear decoder fit to the train set (this decoder is a FIXED measurement instrument — same architecture and training procedure for all backbones to ensure the metric comparison is fair).
5. Measure reconstruction quality with **8 metrics**: {MSE, MAE, ZCR, 1-bit MSE, 2-bit μ-law MSE, PCC, SI-SNR, SDSC}.
6. Rank the 3 backbones by each metric; ask: do MSE and SDSC agree on which backbone is best? Where they disagree, which agrees with the V1 forecasting family-g pattern?

**C-4 pre-registered hypothesis**: SDSC ranking of the 3 backbones identifies the backbone whose reconstructions are most sign-and-magnitude faithful, even when MSE rank is tied or disagrees. ZCR ranking is uninformative (catastrophic).

**C-4 scope**: 3 backbones × 6 dataset-types (3 in-domain + 3 xdomain) × 3 seeds = **54 measurement runs** (each = pretrain + linear decoder fit + metrics eval). Reuses each backbone's natural pretrain.

## C-5 design (loss-in-recon-head)

For each (backbone, dataset, loss ∈ {MSE, SDSC, ZCR}):
1. Pretrain encoder with backbone's **original** loss (identity preserved).
2. Freeze encoder.
3. Add a **post-hoc reconstruction head** (small Linear or MLP, encoder_dim → input_dim).
4. Train the recon head **only** with the chosen reconstruction loss (MSE / SDSC / ZCR) for fixed budget (~50 epochs).
5. Finetune the classification head (jointly with recon head) on labels.
6. Measure test accuracy.

**C-5 pre-registered hypothesis**: Test accuracy across {MSE, SDSC, ZCR} losses in the recon head is within ±2% on in-domain and within ±5% on cross-domain. ZCR may show finite degradation but NOT the catastrophic collapse observed in forecasting (because encoder is already trained; recon head loss has limited downstream effect when encoder is frozen).

**C-5 scope**: 3 backbones × 3 losses × 6 dataset-types × 3 seeds = **162 runs**.

## Acceptance Criteria (v2-locked)

### AC-CL2-1 (frozen cells)
`paper_supplement/protocol/classification_seedcells_v2.json` lists exactly 54 (C-4) + 162 (C-5) = 216 runs with cell_id schema:
- C-4 cell: `c4_{backbone}_{type}_{seeds}` where type ∈ {indomain_Epilepsy, indomain_Gesture, ..., xdomain_SleepEEG__Epilepsy, ...}
- C-5 cell: `c5_{backbone}_{loss}_{type}_{seeds}`

### AC-CL2-2 (C-4 SDSC > ZCR generalization)
For ≥ 6 of the 9 (backbone, dataset-type) combinations, SDSC's ranking of reconstructions correlates with V1 family-g direction (sign-preserving structural damage) with paired bootstrap CI excluding zero against ZCR. Pre-registration locked.

### AC-CL2-3 (C-5 loss-neutrality in head)
For ≥ 7 of the 9 in-domain (backbone, dataset) cells, |acc(SDSC_head) − acc(MSE_head)| ≤ 2%. ZCR head may differ by up to 8% — honest reporting.

### AC-CL2-4 (cross-domain TOST in C-5)
TOST equivalence ±5% accuracy on cross-domain at BH-FDR q=0.05 between MSE_head and SDSC_head, across all (backbone, xdomain-pair). Equivalent ≥ 6/9.

### AC-CL2-5 (page allocation)
Section 7 in AAAI paper: 2 pages text + 1 page appendix. (Up from v1's 1.5p + 0.5p.) Freed budget: V1 Sec 5 table shrunk to per-family rho only; AC-6 table to appendix; original Sec 4 200-cell to appendix.

### AC-CL2-6 (compute tripwire)
Per-cell wallclock cap: 2h pretrain + 1h finetune. If a cell exceeds, subset its source dataset (50k samples for SleepEEG, full retained for smaller).

### AC-CL2-7 (cross-backbone fairness)
All backbones use identical: (a) seed list {42, 123, 2024}, (b) data adapter (consume same .pt format), (c) linear decoder architecture in C-4 (encoder_dim → input_dim, no hidden), (d) post-hoc head architecture in C-5 (Linear or 2-layer MLP — pre-registered choice).

### AC-CL2-8 (honest backbone provenance)
Paper Section 7 must name backbones precisely:
- "SimMTM-Cls" (not "TFC") — note that SimMTM_Classification's class is named TFC but the architecture is SimMTM
- "TF-C (Zhang et al., NeurIPS 2022)" for the real TF-C
- "TS2Vec (Yue et al., AAAI 2022)"

## Scope budget (~10 days compute, single RTX 6000 Ada)

| Phase | Activity | GPU-h | Wallclock |
|---|---|---|---|
| Now-Day3 | C-4 + C-5 infra (wrappers, recon head, decoder, evaluation pipeline) | 0 (CPU dev) | 3 days |
| Day1-3 | SimMTM-Cls C-5 sweep (existing 54-run sweep is already C-5 in-domain + cross-domain) | ~70 GPU-h | 2 days (running) |
| Day3-6 | TF-C pretrain on 6 dataset-types + post-hoc head sweep | ~80 GPU-h | 3 days |
| Day6-9 | TS2Vec pretrain on 6 dataset-types + post-hoc head sweep | ~80 GPU-h | 3 days |
| Day9-11 | C-4 measurement on all 3 backbones × 6 dataset-types × 3 seeds | ~10 GPU-h | 1 day |
| Day11-13 | Unified analysis + Section 7 draft | 0 | 2 days |

Total ~14 days from today (2026-06-19). Comfortably inside AAAI deadline 2026-08-15 (~8 weeks remaining).

## Falsification gates (honest reporting required)

| Observation | Required reporting |
|---|---|
| AC-CL2-2 fails (SDSC ≯ ZCR on <6 backbone-data combos) | "SDSC's discrimination power is encoder-dependent; the V1 forecasting pattern does not fully transfer to all SSL frameworks" |
| AC-CL2-3 fails (loss-in-head differs > 2% on > 2 cells) | "Loss-neutrality holds in encoder pretraining but tightens when applied to post-hoc reconstruction" |
| TF-C reconstruction head saturates at trivial output | "TF-C's frequency contrastive objective does not preserve enough time-domain information for non-trivial reconstruction" — interesting null result |
| TS2Vec hierarchical embedding incompatible with linear decoder | Switch to 2-layer MLP decoder; document choice |

## References

- v1 protocol (superseded): `paper_supplement/protocol/classification_protocol.md`
- v1 seedcells (superseded): `paper_supplement/protocol/classification_seedcells.json`
- Forecasting V1 results: `paper_supplement/protocol/V1_results_summary.md`
- AAAI plan: `paper_supplement/AAAI27_metric_validation_plan.md`
- TF-C upstream: `backbones/TFC/code/TFC/` (cloned from mims-harvard/TFC-pretraining)
- TS2Vec upstream: `backbones/TS2Vec/` (cloned from yuezhihan/ts2vec)
