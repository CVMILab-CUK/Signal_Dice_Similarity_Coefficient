#!/usr/bin/env python3
"""AAAI27 Classification Confirmatory sweep driver (Plan B++ AC-CL-1).

Reads paper_supplement/protocol/classification_seedcells.json and dispatches each
(cell_id, seed) combo to SimMTM_Classification/code/main.py with proper
training_mode + pretrain/target dataset args.

In-domain cells: training_mode='pre_train' followed by 'fine_tune' on the same
dataset (TFC's standard in-domain protocol).

Cross-domain cells: pretrain on source dataset, finetune on target dataset.

Results are appended to outputs/classification_results/{cell_id}_seed{seed}.json
and consolidated by analyze_classification.py.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CLS_DIR = REPO_ROOT / "SimMTM_Classification"
CLS_CODE = CLS_DIR / "code"
SEEDCELLS = REPO_ROOT / "paper_supplement" / "protocol" / "classification_seedcells.json"
OUT_DIR = CLS_DIR / "outputs" / "classification_sweep"


def now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def run_one(cell, seed, out_dir, gpu, dry_run=False):
    """Run a single cell (loss_mode, dataset[/transfer], seed)."""
    cell_id = cell["id"]
    loss = cell["loss"]
    log_path = out_dir / f"{cell_id}_seed{seed}.log"
    json_path = out_dir / f"{cell_id}_seed{seed}.json"

    if "source" in cell:
        # cross-domain transfer
        pretrain_ds = cell["source"]
        target_ds = cell["target"]
    else:
        # in-domain: pretrain on target itself (standard TFC pretrain+finetune)
        pretrain_ds = cell["target"]
        target_ds = cell["target"]

    cmd = [
        "/usr/bin/python3", "-u", "main.py",
        "--seed", str(seed),
        "--training_mode", "pre_train",
        "--pretrain_dataset", pretrain_ds,
        "--target_dataset", target_ds,
        "--loss_mode", loss,
        "--finetune_result_file_name", str(json_path.resolve()),
        "--logs_save_dir", str(out_dir.resolve()),
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
           "TMPDIR": "/workspace/tmp"}

    print(f"[{now_iso()}] {cell_id} seed={seed} loss={loss} "
          f"pretrain={pretrain_ds} -> finetune={target_ds}", flush=True)
    if dry_run:
        print(f"  CMD: {' '.join(cmd)}")
        return 0
    with open(log_path, "wb") as f:
        f.write(f"\n=== {now_iso()} CMD: {' '.join(cmd)} ===\n".encode())
        f.flush()
        proc = subprocess.run(cmd, cwd=str(CLS_CODE), stdout=f,
                              stderr=subprocess.STDOUT, env=env)
    return proc.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--in-domain-only", action="store_true",
                    help="Skip cross-domain cells (debugging).")
    ap.add_argument("--cross-domain-only", action="store_true",
                    help="Skip in-domain cells.")
    ap.add_argument("--cells", default=None,
                    help="Comma-separated cell IDs to run (filter).")
    ap.add_argument("--seeds", default=None,
                    help="Comma-separated seeds (default = all from json).")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    status_tsv = out_dir / "run_status.tsv"

    spec = json.loads(SEEDCELLS.read_text())
    cells = []
    if not args.cross_domain_only:
        cells.extend(spec["in_domain_cells"])
    if not args.in_domain_only:
        cells.extend(spec["cross_domain_cells"])
    if args.cells:
        wanted = set(args.cells.split(","))
        cells = [c for c in cells if c["id"] in wanted]

    seeds_filter = None
    if args.seeds:
        seeds_filter = [int(s) for s in args.seeds.split(",")]

    # Resume support: read DONE entries
    done = set()
    if status_tsv.exists():
        with open(status_tsv) as f:
            next(f, None)
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) >= 4 and p[3] == "DONE":
                    done.add((p[1], int(p[2])))
    print(f"Total cells: {len(cells)} unique configs", flush=True)
    print(f"Already DONE in status TSV: {len(done)}", flush=True)

    if not status_tsv.exists():
        with open(status_tsv, "w") as f:
            f.write("timestamp\tcell_id\tseed\tstatus\n")

    idx = 0
    total_runs = sum(
        len([s for s in c["seeds"] if seeds_filter is None or s in seeds_filter])
        for c in cells
    )
    for cell in cells:
        seeds = cell["seeds"]
        if seeds_filter is not None:
            seeds = [s for s in seeds if s in seeds_filter]
        for seed in seeds:
            idx += 1
            if (cell["id"], seed) in done:
                print(f"[{idx}/{total_runs}] SKIP {cell['id']} seed={seed} (DONE)",
                      flush=True)
                continue
            print(f"\n[{idx}/{total_runs}] ===== {cell['id']} seed={seed} =====",
                  flush=True)
            rc = run_one(cell, seed, out_dir, args.gpu, args.dry_run)
            status = "DONE" if rc == 0 else f"FAIL_rc{rc}"
            with open(status_tsv, "a") as f:
                f.write(f"{now_iso()}\t{cell['id']}\t{seed}\t{status}\n")

    print(f"\n[{now_iso()}] Sweep complete. Status: {status_tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
