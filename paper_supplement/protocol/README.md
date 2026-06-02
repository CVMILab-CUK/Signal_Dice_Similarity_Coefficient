# paper_supplement/protocol/

Pre-registration artifacts for the AAAI27 SDSC metric validation plan v5.

Status: pending approval (see ../AAAI27_metric_validation_plan.md)

## Contents (populated per AC-1, AC-5, AC-10, AC-11)

- `PREREG_HASH`        — git commit hash + ISO timestamp at protocol freeze
- `V1_protocol.md`     — 10-family pair construction + bracketed predictions (AC-1)
- `V1_data_inventory.md` — EEG/ECG/ETT/Weather data audit (AC-10)
- `ac6_seedcells.json` — 12 unique configs × N=5 seeds frozen list (AC-6 via Δ8)
- `V3_recoverability.md` — pretrain reconstruction tensor audit (AC-4)

## Pre-registration gate

V1 analysis notebook must read `PREREG_HASH` frontmatter and abort if
uncommitted. Notebook can iterate freely; protocol/ subdir is frozen at
week-1 commit-hash gate.
