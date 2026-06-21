# C2 Integration Plan — End-to-End Forecasting Models

> Status: **research-only prep** for next session. Repos NOT cloned/modified
> yet. The 4 models (QDF, TQNet, Fredformer, PDF) are end-to-end forecasting
> models — no pretrain phase. SDSC integrates as a **direct forecasting loss**
> replacing MSE in the training criterion. This is a different experimental
> setting from C1 (where SDSC is the pretrain reconstruction loss).

## Common integration pattern (3 of 4 models)

TQNet, Fredformer, and PDF all build on **Time-Series-Library** infrastructure
(`exp/exp_long_term_forecasting.py` style). Integration is uniform:

1. Clone repo
2. Locate the training criterion — typically
   `criterion = nn.MSELoss()` inside `_select_criterion()` or `train()` body.
3. Replace with a small loss factory that maps `--loss_mode` to one of our 8
   losses (mse / sdsc / hybrid / dtw / pcc / snr / zcr / dilate).
4. After test() computes `mae, mse, rmse, mape, mspe`, also compute SDSC on
   the test forecasts (same code we added to our `exp_simmtm.py:test()`).
5. Write to `outputs/test_results/{data}/{data}_{loss}_score.txt` in our
   `"{seq}->{pred}, mse, mae, sdsc"` 4-column format, so our existing
   `analyze_multi_v2.py` picks it up.
6. Add to multi_dataset_sweep.py MODELS list — driver invokes their `run.py`
   with `--model TQNet` etc.

## Per-model notes

### 1. TQNet (ICML 2025) — EASIEST integration
- Repo: `ACAT-SCUT/TQNet`
- Built on Time-Series-Library + scripts in `./scripts/TQNet/`
- Lightweight (~1M params), single-layer attention + MLP
- Loss replacement: swap `nn.MSELoss` in their `exp/exp_long_term_forecasting.py`
- Expected runtime: similar to PatchTST (fast). ETTh1 ~25 min/run.
- Datasets supported: ETT family, Weather, ECL, Traffic, plus periodic data sets.

### 2. Fredformer (KDD 2024) — EASY
- Repo: `chenzrg/fredformer`
- Built on PyTorch, frequency-debiased patch transformer
- Key params: `--patch_len 16 --cf_dim 48 --cf_depth 2 --cf_heads 6 --cf_mlp 128`
- Loss replacement: same pattern as TQNet
- Expected runtime: similar to PatchTST
- Caveat: ECL/Traffic uses Nystrom variant by default (`--use_nys 1`). Keep
  the default for fair comparison with their published numbers.

### 3. PDF (ICLR 2024) — EASY
- Repo: `Hank0626/PDF`
- Built on Time-Series-Library
- Two input-length variants: `./scripts/PDF/336/{dataset}.sh` and `./scripts/PDF/720/{dataset}.sh`
- For paper-comparison parity, run input_length=336 (matches our seq_len=96 ETT
  but their best results are at 720; we should pick ONE and stick with it).
- Loss replacement: same pattern.
- Expected runtime: similar to PatchTST.

### 4. QDF (recent, Quadratic Direct Forecast) — TRICKY
- Repo: `Master-PLC/QDF`
- **Different architecture**: meta-learning that LEARNS a task-adaptive loss
  function via meta-train + meta-test + inner_loop. The core novelty is the
  `CovarianceMatrix` class implementing a covariance-based learned loss.
- **Two integration options**:
  - **(a) Replace QDF's learned loss with our SDSC** — defeats QDF's
    contribution. Just uses QDF's encoder backbone trained with our 8 losses
    directly (skipping their meta-learning entirely). Cleanest comparison;
    QDF reduces to "yet another forecasting backbone" in our table.
  - **(b) Compose: keep QDF meta-learning, but ALSO add SDSC as one of the
    candidate losses in their meta-task pool**. Much harder; requires
    understanding their inner_loop and meta_train/meta_test phases.
- **Recommendation**: Go with (a) for paper parity. Note in paper that
  "QDF backbone trained with fixed loss (our 8 modes) rather than QDF's
  meta-learned loss, for fair comparison across backbones."
- Expected runtime: slow (meta-learning takes more compute even without it,
  since QDF's encoder is larger). Probably ~60-90 min/run on ETTh1.

## Compute estimate for C2 (all 7 datasets × 4 models × 8 losses × 1 seed)

Per model assuming similar to PatchTST/iTransformer:
- ETTh1, ETTh2: 8 losses × ~25 min = 3-4h each
- ETTm1, ETTm2: 8 × ~80 min = 11h each
- Weather: 8 × ~40 min = 5h
- ECL: 8 × ~10h = 80h
- Traffic: 8 × ~30h = 240h
- Total per model: ~350h (3-4 models with slight variation)
- All 4 models: **~1400 GPU-h ≈ 58 days** sequential

QDF likely 1.5× slower due to backbone size → add ~150h.

C2 total: **~1500-1700h** (similar order of magnitude to C1).

If C2 runs after C1 completes, total project = ~33d (C1 with 4090) + ~58d (C2 sequential) = ~91 days.

**Suggested mitigation for C2**:
- Limit to 4 datasets (ETTh1, ETTh2, ETTm1, Weather) and 1 seed for first pass → ~280 runs ≈ 25 days
- Add ETT family + Weather first; ECL/Traffic for C2 only as supplementary
- OR keep 4090 reserved for C2's heavy datasets after Traffic on 4090 finishes

## Integration steps (next session work order)

1. Clone TQNet, Fredformer, PDF, QDF into `external_models/` subfolder.
2. Per repo: add `utils/sdsc_loss.py` that re-exports our loss factory (or
   `pip install -e .` our SimMTM_Forecasting/utils as a local package).
3. Patch each repo's `exp_long_term_forecasting.py`:
   - Add `--loss_mode` arg
   - Replace `criterion = nn.MSELoss()` with `criterion = make_loss(args.loss_mode)`
   - Add SDSC computation in test() — same snippet as our exp_simmtm.py.
4. Add to `datasets_config.py` MODELS list (or write per-model wrappers).
5. Update `multi_dataset_sweep.py` to route TQNet/Fredformer/PDF/QDF to their
   own external repo's `run.py` rather than SimMTM_Forecasting/run.py.
6. Smoke test each (ETTh1, mse, 1 epoch).
7. Launch sweep (large compute commitment).

## User decisions (locked in 2026-05-21)

- **(a) QDF**: **backbone-only** — skip QDF's meta-learning entirely, train the
  QDF encoder with our 8 losses just like any other backbone. Paper text:
  "QDF backbone trained with fixed loss for fair multi-backbone comparison
  (their meta-learned loss is out of scope)."
- **(b) PDF input length**: **336** (their `./scripts/PDF/336/{dataset}.sh`).
  We run PDF with seq_len=336 / pred_len=96, distinct from the seq_len=96
  baseline of the other 3 models. Note this in the paper table footnote.
- **(c) C2 scope**: **4 models × 7 datasets × 8 losses × seed=2023 = 224 runs**.
  Full coverage; no dataset cut.
- **(d) C2 timing**: **start C2 the moment C1 ETT/Weather are done** (overlap
  C2 ETT/Weather with C1 ECL — both nodes useful in parallel). C2 ECL/Traffic
  start when 4090 finishes Traffic (Plan A).

## Execution sequencing

```
Day  0:  C1 launched (this session). 6000-Ada runs ETT family + Weather + ECL.
         4090 runs C1-Traffic only (Plan A).
Day ~14: C1 ETT/Weather done on 6000-Ada. C2 kicks off on 6000-Ada starting
         with TQNet (easiest) on ETT/Weather. C1 ECL continues on 6000-Ada.
         (Single GPU — sequential. So actually C2 has to wait for C1 ECL.)
Day ~27: C1 ECL done on 6000-Ada. C2 starts on 6000-Ada with ETT/Weather.
Day ~33: 4090 finishes C1-Traffic.
         4090 starts C2-Traffic? Or C2 ECL on 4090 while main does smaller
         C2 datasets. Decision deferred until that point.
```

In practice with 1 main GPU, "C1 ETT/Weather finish → start C2" requires C1 to
fully pause ECL/Traffic. Since ECL is heavy, the realistic sequence is:
1. C1 main runs ETT/Weather/ECL sequentially (24–27 days).
2. C2 main starts after C1 main finishes ECL.
3. 4090 runs C1-Traffic the whole time, then C2 heavies after.

For overlap to actually save days, we'd need C2 on the 4090 while C1-Traffic
runs there too — that means BOTH sweeps competing for the same 24GB. Probably
not feasible. So effective parallelism is "C1-main on 6000-Ada / C1-Traffic on
4090" then "C2-main on 6000-Ada / C2-heavies on 4090".
