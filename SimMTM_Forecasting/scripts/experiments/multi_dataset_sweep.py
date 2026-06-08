#!/usr/bin/env python3
"""Multi-dataset × multi-model × multi-loss sweep driver.

Iterates over (dataset, model, loss, seed) and shells out to run.py with the
right --flags. Resumable via per-run status TSV; failed runs can be retried by
re-running the script.

Usage:
    /usr/bin/python3 scripts/experiments/multi_dataset_sweep.py \\
        [--datasets ETTh1,ETTh2,...] \\
        [--models SimMTM,PatchTST,iTransformer] \\
        [--losses mse,sdsc,...] \\
        [--seeds 2023] \\
        [--out-dir outputs/experiments/multi_sweep_v1] \\
        [--gpu 0]

Defaults: all datasets, all 3 models, all 8 losses, seed=2023.
"""

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Allow running from any cwd.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from datasets_config import DATASETS, LOSS_MODES, MODELS, emit_run_args  # noqa: E402

REPO_ROOT = SCRIPT_DIR.parent.parent  # SimMTM_Forecasting/

# (dataset, model) pairs to skip entirely. iTransformer on Traffic has
# enc_in=862 x d_model=128 in its Pooler_Head Linear(enc_in*d_model, .../2),
# producing ~6B parameters — won't fit a single 48GB GPU even at batch=1.
SKIP_COMBOS = {("Traffic", "iTransformer")}

# (dataset, loss) pairs to skip across all models. DILATE on large-channel
# datasets is infeasible:
#   - ECL × DLinear × dilate measured 26,100 s/epoch (7.25 h/epoch -> 15 days/cell)
#   - Traffic × PatchTST × dilate diverged to NaN/Inf in early epochs
# Reported in the paper as a baseline-scope limitation, not a SDSC limitation.
SKIP_DATASET_LOSS = {("ECL", "dilate"), ("Traffic", "dilate")}


def now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_status(status_file):
    done = set()
    if not status_file.exists():
        return done
    with open(status_file) as f:
        next(f, None)  # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 5 and parts[4] == "DONE":
                done.add((parts[1], parts[2], parts[3]))  # (dataset, model, loss)
    return done


def append_status(status_file, dataset, model, loss, status):
    new = not status_file.exists()
    with open(status_file, "a") as f:
        if new:
            f.write("timestamp\tdataset\tmodel\tloss\tstatus\n")
        f.write(f"{now_iso()}\t{dataset}\t{model}\t{loss}\t{status}\n")


def run_phase(args_list, env, log_file, append=False):
    """Invoke run.py with the given flags; tee output to log_file."""
    cmd = [env.get("PYTHON", "/usr/bin/python3"), "-u", "run.py"] + args_list
    mode = "ab" if append else "wb"
    with open(log_file, mode) as f:
        f.write(("\n" + "=" * 60 + f"\n[{now_iso()}] CMD: {' '.join(cmd)}\n" + "=" * 60 + "\n").encode())
        f.flush()
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=f, stderr=subprocess.STDOUT,
                              env={**os.environ, **env})
    return proc.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(DATASETS.keys()))
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--losses", default=",".join(LOSS_MODES))
    ap.add_argument("--seeds", default="2023")
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "outputs" / "experiments" / "multi_sweep_v1"))
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--skip-pretrain-if-ckpt-exists", action="store_true",
                    help="If pretrain ckpt exists at the expected path, skip pretrain and run finetune only.")
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    models   = [m.strip() for m in args.models.split(",")   if m.strip()]
    losses   = [l.strip() for l in args.losses.split(",")   if l.strip()]
    seeds    = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    status_file = out_dir / "run_status.tsv"
    done = read_status(status_file)

    env = {"CUDA_VISIBLE_DEVICES": args.gpu, "PYTHON": "/usr/bin/python3"}
    total = len(datasets) * len(models) * len(losses) * len(seeds)
    cnt_done_at_start = len([k for k in done if k[0] in datasets and k[1] in models and k[2] in losses])
    print(f"Total runs scheduled: {total} (already DONE in status file: {cnt_done_at_start})")
    print(f"Datasets: {datasets}")
    print(f"Models:   {models}")
    print(f"Losses:   {losses}")
    print(f"Seeds:    {seeds}")
    print(f"GPU:      {args.gpu}")
    print(f"Out dir:  {out_dir}")

    idx = 0
    for seed in seeds:
        for dataset in datasets:
            for model in models:
                # Per-model checkpoint path (saved by exp_simmtm.py:67 / our US-005 fix).
                ckpt_path = REPO_ROOT / "outputs" / "pretrain_checkpoints" / DATASETS[dataset]["data"]
                # Loss_mode is appended below.
                for loss in losses:
                    idx += 1
                    if (dataset, model) in SKIP_COMBOS:
                        if loss == losses[0]:
                            print(f"[{idx}/{total}] SKIP {dataset}/{model} (memory-infeasible combo)")
                        continue
                    if (dataset, loss) in SKIP_DATASET_LOSS:
                        print(f"[{idx}/{total}] SKIP {dataset}/{model}/{loss} (DILATE infeasible on large-T datasets)")
                        continue
                    key = (dataset, model, loss)
                    if key in done:
                        print(f"[{idx}/{total}] SKIP {dataset}/{model}/{loss}/seed={seed} (DONE)")
                        continue

                    print(f"\n[{idx}/{total}] ===== {dataset} | {model} | {loss} | seed={seed} =====")
                    log_file = out_dir / f"{dataset}_{model}_{loss}_seed{seed}.log"

                    pretrain_args = emit_run_args(dataset, "pretrain", model, loss, seed)
                    finetune_args = emit_run_args(dataset, "finetune", model, loss, seed)

                    expected_ckpt = ckpt_path / loss / "ckpt_best.pth"
                    # Pretrain phase
                    skip_pretrain = args.skip_pretrain_if_ckpt_exists and expected_ckpt.exists()
                    if skip_pretrain:
                        print(f"  pretrain SKIP (ckpt exists at {expected_ckpt})")
                    else:
                        rc = run_phase(pretrain_args, env, log_file, append=False)
                        if rc != 0:
                            print(f"  pretrain FAILED (rc={rc}); see {log_file}")
                            append_status(status_file, dataset, model, loss, "PRETRAIN_FAIL")
                            continue

                    # Finetune + test
                    rc = run_phase(finetune_args, env, log_file, append=True)
                    if rc != 0:
                        print(f"  finetune FAILED (rc={rc}); see {log_file}")
                        append_status(status_file, dataset, model, loss, "FINETUNE_FAIL")
                        continue

                    append_status(status_file, dataset, model, loss, "DONE")
                    print(f"  DONE {dataset}/{model}/{loss}/seed={seed}")

    print(f"\nSweep complete. Status: {status_file}")


if __name__ == "__main__":
    main()
