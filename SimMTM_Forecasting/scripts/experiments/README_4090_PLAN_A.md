# Plan A — Parallelize Traffic on RTX 4090

> **TL;DR**: Run the Traffic sub-sweep (8 losses × 3 backbones × seed=2023 = 24 runs,
> ~800 GPU-h, ≈33 days) on a 4090 box in parallel with the main 6000-Ada sweep,
> cutting total wall-clock from ~58 days to ~33 days. Net saving ≈25 days.

## 0. Portable setup — ONE env variable change

`scripts/experiments/datasets_config.py` reads the data root from
`SDSC_DATA_ROOT` env var (default `/workspace/data/signal/forecasting`). On the
4090 box just `export SDSC_DATA_ROOT=/your/path/to/forecasting` before running
any sweep script. No code modification required.

Quickest path to start on 4090:

```bash
# 1) tar the whole SimMTM_Forecasting/ + dataset on the main machine:
cd /root/jeyoung/codes/Signal_Dice_Similarity_Coefficient
tar czf /tmp/sdsc_4090_bundle.tgz \
    SimMTM_Forecasting/ \
    --exclude='SimMTM_Forecasting/outputs/pretrain_checkpoints/*' \
    --exclude='SimMTM_Forecasting/outputs/logs/*' \
    --exclude='SimMTM_Forecasting/outputs/experiments/*' \
    --exclude='SimMTM_Forecasting/outputs/_legacy_simmtm_sweep_v1/*' \
    --exclude='**/__pycache__'

# 2) copy data (only Traffic needed) — alternative if you already have it on 4090:
tar czf /tmp/sdsc_traffic_data.tgz -C /workspace/data/signal/forecasting traffic/

# 3) on 4090:
mkdir -p ~/sdsc_4090
tar xzf sdsc_4090_bundle.tgz -C ~/sdsc_4090
mkdir -p ~/sdsc_data
tar xzf sdsc_traffic_data.tgz -C ~/sdsc_data

# 4) one-time env setup + launch:
export SDSC_DATA_ROOT=~/sdsc_data
cd ~/sdsc_4090/SimMTM_Forecasting
bash scripts/experiments/traffic_only_4090.sh
```

That's it. The default path inside `datasets_config.py` is **overridden** by
the env var, so the script works wherever the data lives.

---

## 1. Why Traffic only?

- Traffic alone is ≈57% of the main queue compute (862 channels × batch_size=4 × seq_len=96).
- 4090 (24 GB) has just enough memory: Traffic config uses batch_size=4 with `--use_amp`,
  estimated peak ≈12–16 GB. Drop to batch_size=2 if OOM hits.
- Putting BOTH ECL + Traffic on 4090 saves only ~12 days (4090 becomes the new bottleneck);
  Traffic-only is the sweet spot.

## 2. Files to transfer (4090 box)

```bash
# From main (6000-Ada) machine:
rsync -avzh --info=progress2 \
    /root/jeyoung/codes/Signal_Dice_Similarity_Coefficient/SimMTM_Forecasting/ \
    user@4090host:/path/to/Signal_Dice_Similarity_Coefficient/SimMTM_Forecasting/

# Also data (Traffic only is enough, but mirroring the rest is fine too):
rsync -avzh --info=progress2 \
    /workspace/data/signal/forecasting/traffic/ \
    user@4090host:/workspace/data/signal/forecasting/traffic/
```

## 3. Environment on the 4090 box

```bash
# Python 3.12 (system) — same as main machine:
sudo apt-get install -y python3 python3-pip
pip install --break-system-packages torch torchvision  # match main's torch>=2.6
pip install --break-system-packages \
    seaborn sktime tensorboardX numba pysdtw einops reformer-pytorch \
    sympy pandas scikit-learn matplotlib
```

(Replace `/usr/bin/python3` references in the experiment scripts with the
correct path if the venv on the 4090 differs.)

## 4. Launch on the 4090

```bash
cd /path/to/Signal_Dice_Similarity_Coefficient/SimMTM_Forecasting
bash scripts/experiments/traffic_only_4090.sh
```

The shell wrapper invokes:

```bash
/usr/bin/python3 scripts/experiments/multi_dataset_sweep.py \
    --datasets Traffic \
    --models SimMTM,PatchTST,iTransformer \
    --losses mse,sdsc,hybrid,dtw,pcc,snr,zcr,dilate \
    --seeds 2023 \
    --gpu 0 \
    --out-dir outputs/experiments/multi_sweep_v1_4090
```

It writes:
- per-run logs: `outputs/experiments/multi_sweep_v1_4090/Traffic_*.log`
- status TSV: `outputs/experiments/multi_sweep_v1_4090/run_status.tsv`
- pretrain ckpts: `outputs/pretrain_checkpoints/Traffic/{loss}/ckpt_best.pth`
- forecasting scores: `outputs/test_results/Traffic/Traffic_{loss}_score.txt`
  (one line per (model, loss) run, in execution order)

Resumable: re-running the same command after a crash skips DONE entries.

## 5. After 4090 finishes — merge results back

```bash
# On 4090 box: collect just the score files + status
tar czf traffic_results_4090.tgz \
    outputs/test_results/Traffic/ \
    outputs/experiments/multi_sweep_v1_4090/run_status.tsv
scp traffic_results_4090.tgz user@main:/tmp/

# On main: merge into the main sweep tree
cd /root/jeyoung/codes/Signal_Dice_Similarity_Coefficient/SimMTM_Forecasting
tar xzf /tmp/traffic_results_4090.tgz
# Append the 4090's run_status entries into the main TSV
cat outputs/experiments/multi_sweep_v1_4090/run_status.tsv \
    | tail -n +2 \
    >> outputs/experiments/multi_sweep_v1/run_status.tsv

# Re-run analyzer with merged data
/usr/bin/python3 scripts/experiments/analyze_multi_v2.py
```

## 6. On the main 6000-Ada — skip Traffic so it doesn't run there too

You need to RESTART the main sweep with `--datasets` excluding Traffic.
The current sweep was launched without that flag and will eventually hit Traffic
(after ~25 days). Restarting now uses the same `run_status.tsv` so any already-DONE
cells are skipped — only the in-progress cell restarts.

```bash
# Kill current main sweep
PID=$(cat /tmp/multi_sweep_v1.pid)
kill $PID 2>/dev/null
sleep 5
kill -9 $PID 2>/dev/null || true
# Wait for the GPU child process to exit
pkill -f "run.py --task_name pretrain --root_path /workspace/data/signal/forecasting/" 2>/dev/null

# Relaunch with Traffic excluded
nohup /usr/bin/python3 scripts/experiments/multi_dataset_sweep.py \
    --datasets ETTh1,ETTh2,ETTm1,ETTm2,Weather,ECL \
    > /tmp/multi_sweep_v1.log 2>&1 &
echo $! > /tmp/multi_sweep_v1.pid
```

The relaunch will re-do the in-progress ETTh1/SimMTM/sdsc cell from epoch 0, but
that's only ~50 min lost.

## 7. Monitoring (both nodes)

```bash
# On main: live progress
ps -p $(cat /tmp/multi_sweep_v1.pid) -o pid,etime
cat outputs/experiments/multi_sweep_v1/run_status.tsv
tail -f $(ls -t outputs/experiments/multi_sweep_v1/*.log | head -1)

# On 4090: same files but in multi_sweep_v1_4090/
ps -p $(cat /tmp/traffic_only_4090.pid) -o pid,etime
cat outputs/experiments/multi_sweep_v1_4090/run_status.tsv
tail -f /tmp/traffic_only_4090.log
```

## 8. When everything is done

Run the analyzer (it auto-includes Traffic if the merged tree has the rows):

```bash
/usr/bin/python3 scripts/experiments/analyze_multi_v2.py
cat outputs/experiments/multi_sweep_v1/results_table.md
```

## 9. Expected timeline

| Node | Workload | ETA |
|---|---|---|
| 6000 Ada (main) | ETTh1, ETTh2, ETTm1, ETTm2, Weather, ECL (144 runs) | ≈25 days |
| 4090 (parallel) | Traffic (24 runs) | ≈33 days |
| **Wall-clock** | max(25, 33) | **≈33 days** |
| (Saving vs. sequential) | | **≈25 days** |
