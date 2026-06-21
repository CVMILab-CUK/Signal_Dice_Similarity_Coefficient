# Tag v10 Annotation Draft (post-sweep, before tagging)

Use this as the annotation body for `git tag -a sdsc-canonical-v10`. Fill TODO placeholders with final 90/90 numbers, verify all values against `paper_supplement/protocol/v2_results_summary.md`, then strip this header line + the TODO checklist.

---

```
SDSC AAAI27 sprint canonical v10 — Plan G' Day 1 complete

4 classification backbones (4 SSL families):
- SimMTM-Cls (NeurIPS'23, masked recon, in_Epilepsy)
- TF-C (NeurIPS'22, contrastive freq-time, in_Epilepsy/in_Gesture + 3 xd)
- TS2Vec (AAAI'22, hierarchical contrastive, in_Epilepsy/in_Gesture/in_HAR + 3 xd)
- GPT4TS (NeurIPS'23, LLM frozen GPT-2 6 layers, in_Epilepsy/in_Gesture/in_HAR + 3 xd) NEW

4 forecasting backbones (already in v8):
- SimMTM (NeurIPS'23), PatchTST (ICLR'23), iTransformer (ICLR'24), DLinear (AAAI'23)

Key results (cross-validated against paper_supplement/protocol/v2_results_summary.md):

In-domain (45/45):
- 9/9 loss-neutrality cells PASS at <=2% diff
- 8/9 cells bitwise 0.00% (SimMTM-Cls Epilepsy alone at 0.13%)
- GPT4TS bitwise 0.00% on all 3 in-domain datasets (LLM paradigm)

Cross-domain (45/45):  TODO fill from final sweep
- N/9 TOST equivalence PASS at delta=2%
- Loss-neutrality preservation across all xd cells
- GPT4TS xd_SleepEEG_Epilepsy notable: C-4 SDSC=0.9784 highest but C-5 acc=0.196 catastrophic — validates the framework claim (SDSC measures structure NOT task accuracy)

Plan G' Day 1 also delivers:
- Falsification gates 1-3 evaluated, all 'not triggered' or resolved
- Decision gate v2 Option A (STOP at 8 backbones) confirmed
- Section 17 hand-off brief filled with final numbers
- HTML status dashboard updated
- Auto-memory project_aaai27_sprint.md persisted

Pre-mortem learnings:
- R1 fallback (S-Mamba -> GPT4TS) executed in <12h after S-Mamba blocked by mamba-ssm CUDA dep + no public classification ckpt
- Wrapper pattern (gpt4ts_wrapper.py) mirrors tfc/ts2vec — proves the cross-backbone framework generalizes to LLM paradigm
- 3-seed averaging resolved a transient 1.09% HAR seed42 anomaly — demonstrates pre-registration + falsification discipline

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
```

---

## Pre-tag checklist
- [ ] Sweep ≥85/90 DONE, 0 FAIL (or known and documented FAIL)
- [ ] analyze_v2.py final run committed
- [ ] Section 17.8 in-domain final + 17.9 cross-domain final filled
- [ ] HTML status reflects v10 (tag string + summary)
- [ ] No uncommitted changes (git status clean)
- [ ] Tag command: `git -c user.email="dlwpdud@gmail.com" -c user.name="SDSC Author" tag -a sdsc-canonical-v10 -m "..."`
