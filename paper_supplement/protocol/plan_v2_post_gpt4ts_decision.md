# Plan v2 — Post-GPT4TS Decision Gate

**Status**: pending approval (ralplan consensus 2026-06-21)
**Context**: After Plan G' R1 (GPT4TS pivot), should we add yet another 2024-25 backbone?
**Default**: Option A (STOP at 8 backbones) — pending user override

## Principles
1. Honest scope > completeness theater
2. Diversity of paradigm > count of backbones
3. 8주 안에 paper writing이 핵심 risk
4. C-4/C-5 framework 적용 가능성 사전 검증 필수

## Decision Drivers
1. Reviewer perception of 2024-25 representation (GPT4TS 2023 LLM family covers post-2023 modern paradigm)
2. Compute budget (1 GPU RTX 6000 Ada 48GB, 8주 deadline AAAI27 2026-08-15)
3. Story coherence (SDSC as metric — diminishing returns on backbone count)

## Viable Options

| Option | What | Risk |
|---|---|---|
| **A. STOP (Recommended)** | 8 backbones (4F + 4C inc GPT4TS) + paper writing | LOW |
| B. + TimeMixer (ICLR'24) | MLP-Mixer family | MED |
| C. + S-Mamba retry | Mamba SSM family | HIGH (R1 already invalidated by mamba-ssm + no public ckpt) |
| D. B + future-work commitment | TimeMixer now, S-Mamba as future work | MED |

## Invalidation Rationale
- Option C: Plan G' pre-mortem already documented S-Mamba blockers (no classification ckpt + mamba-ssm CUDA dependency). Re-attempting carries same risk.
- Moirai/Chronos/TimesFM: forecasting-only, no classification track applicable.
- TimeMachine (ICLR'25): classification path unverified by upstream authors.
- "QDF" (user-mentioned): citation unverified; cannot integrate unverified model.

## Decision Gate (commit point)

After GPT4TS sweep ≥85/90 DONE:
1. Run `analyze_v2.py` for final unified summary.
2. Inspect 2 metrics:
   - **In-domain loss-neutrality (AC-CL2-3)**: ≥7/8 cells PASS at ≤2% diff?
   - **Cross-domain TOST equivalence (AC-CL2-4)**: ≥4/6 cells PASS?
3. **If BOTH pass → Option A** (STOP, tag v10, paper writing).
4. **If EITHER fails on >2 cells → Option B trigger** (separate ralplan for TimeMixer).
5. **If "QDF" user clarification arrives with verified citation → re-evaluate.**

## Falsification Conditions
- Falsification 1: If sweep shows >3 FAIL cells → not Option A path; investigate root cause first.
- Falsification 2: If TS-2024 reviewer simulation (devil's-advocate writeup) returns "outdated benchmark" verdict → Option B trigger regardless of sweep result.
- Falsification 3: **GPT4TS in_HAR seed42 already shows 1.09% MSE-vs-SDSC diff (under 2% threshold but first non-zero)**. If 3-seed average remains >1%, report as nuanced finding: "GPT4TS loss-neutrality holds within ≤2% tolerance but is NOT bitwise like contrastive backbones — suggests SSL framework type matters for exact head-decoupling." This *strengthens* the differentiation-instrument framing rather than weakening it.

## Acceptance Criteria for STOP path
- [ ] Sweep ≥85/90 DONE with 0 FAIL
- [ ] analyze_v2 produces v2_results_summary.md with 8-backbone table
- [ ] Section 17.8-9 final fill in hand-off brief (replacing interim)
- [ ] Commit + tag `sdsc-canonical-v10`
- [ ] paper_supplement/SDSC_AAAI27_status.html updated to reflect final coverage
- [ ] Section 18 "Future work" line includes TimeMixer/S-Mamba/Mamba-family as deferred

## ADR

- **Decision**: Option A (STOP at 8 backbones) is the default; B is a contingent fallback.
- **Drivers**: Deadline risk > completeness; paradigm diversity already adequate (5 families).
- **Alternatives considered**: B/C/D rejected per invalidation rationale above.
- **Why chosen**: GPT4TS (NeurIPS'23, LLM paradigm) is the boundary case for "modern" — 2023 LLM-for-TS marks the post-2023 paradigm shift, not a 2022-23 incremental. Adding TimeMixer (2024) would shift the perception from "principled paradigm coverage" to "completeness theater".
- **Consequences**: Reviewer "no 2025 model" criticism possible. Mitigation: Section 18 explicitly addresses with "Future work: post-2024 SSM and foundation models pending public classification checkpoints."
- **Follow-ups**: Monitor GPT4TS sweep completion; trigger gate evaluation; if Option B triggered, separate ralplan for TimeMixer integration.

## Verification Steps
1. `cat SimMTM_Classification/outputs/v3_gpt4ts_sweep/run_status.tsv | awk -F'\t' '$2=="DONE"' | wc -l` → ≥85
2. `python3 paper_supplement/scripts/analyze_v2.py` → produces summary
3. `grep -c "PASS" paper_supplement/protocol/v2_results_summary.md` → check AC-CL2-3/4 counts
4. Gate decision logged in this file (append section "Gate Decision YYYY-MM-DD").
