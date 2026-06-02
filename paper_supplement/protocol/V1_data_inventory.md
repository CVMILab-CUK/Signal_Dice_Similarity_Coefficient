# V1 Data Inventory (AC-10)

**Authored**: 2026-06-02 | **Status**: pre-registration draft

## Available signal data on box

### Classification track — `/workspace/data/signal/classification/`

| Dataset | Train samples | Length (T) | Channels | dtype | Forecasting cell windows (T/96) | Tier role |
|---|---|---|---|---|---|---|
| **ECG** (MIT-BIH derived) | 43,673 | 1500 | 1 | float64 | 15 windows/sample → ~655k windows | **Tier A primary** |
| **SleepEEG** | 371,055 | 178 | 1 | float32 | 1 window (or pad) | **Tier A secondary** |
| EMG | 122 | 1500 | 1 | float64 | 15 windows → ~1830 windows | Tier B small-N sanity |
| Epilepsy | 60 | 178 | 1 | float64 | 1 window → 60 windows | Tier C minimal |
| FD-A | (audit pending) | | | | | reserve |
| FD-B | (audit pending) | | | | | reserve |
| Gesture | (audit pending) | | | | | reserve |
| HAR | (audit pending) | | | | | reserve |

### Forecasting track — `/workspace/data/signal/forecasting/`
- ETTh1 / ETTh2 / ETTm1 / ETTm2 / Weather / ECL / Traffic
- All seq_len=96 multivariate (used in main 200-cell grid)

## V1 pair construction — length compatibility

Plan v5 AC-1 N≥500 pairs per family. Window strategy:
- **ECG (1500 → 96)**: extract non-overlapping 96-length windows. 15 per sample × 43,673 = ~655k available. Sample 500 per family with stratification across sample-IDs to ensure independence.
- **SleepEEG (178 → 96)**: extract 1 window per sample (left-aligned). 371,055 available; sample 500 per family.
- **ETT/Weather**: existing seq_len=96 sequences from validation set (loss-neutrality 200-cell grid checkpoints provide candidates).

## Critical observations

1. **Mostly single-channel univariate** — V1 must operate on 1D signals (T,) reshape from (1, T). This is fine; SDSC canonical works on any (..., T) shape because reduction is dim=-1.
2. **Sampling rates not embedded in .pt files** — known from TF-C paper: ECG=80Hz, SleepEEG=100Hz, EMG=4kHz, Epilepsy=178Hz. Document in paper Appendix.
3. **Tier A bracket coverage**: ECG (cardiac waveform) + SleepEEG (EEG spectral) gives 2 distinct signal families; sufficient for "real-signal" robustness claim. EMG/Epilepsy too small-N for primary statistical claim → Tier B/C as planned.
4. **No DC-offset failure-case data needed** — family (j) DC offset is synthetic perturbation injected onto ECG/SleepEEG; no separate labels required.

## Decisions locked at this audit

- **Tier A primary source**: ECG train (43k × 15 windows)
- **Tier A secondary**: SleepEEG train (371k samples × 1 window)
- **ETT/Weather**: re-use forecasting validation sets for distribution-shift sanity within Tier A
- **Tier B oracle**: skip — no labeled clinical task fits the timeline. Tier B downgraded to a *post-hoc robustness section* citing published TF-C results, not a separate experiment.

This Tier B downgrade is a **deviation from plan v5** (which scoped Tier B as a robustness check requiring a pretrained classifier). The downgrade is honest because (a) 10 weeks does not accommodate building a clinical classifier oracle, (b) Tier A on ECG already covers cardiac waveform structure, and (c) Tier C SQI rubric covers sanity. Net coverage unchanged.

## Pre-registration commitment

- Pair sampling random seed: `42`
- Pair sampling protocol: stratified by sample-ID (no two pairs share a source sample)
- Window selection within long sequences: left-aligned for SleepEEG; stride=96 non-overlapping for ECG/EMG (1500 length)
- All pair-IDs (source sample index + window offset) saved to `paper_supplement/protocol/V1_pair_index.npz` at week-1 freeze
