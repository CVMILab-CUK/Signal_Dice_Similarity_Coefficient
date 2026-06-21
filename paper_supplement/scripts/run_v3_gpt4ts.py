#!/usr/bin/env python3
"""AAAI27 protocol Section 17 (G'-GPT4TS pivot) sweep driver.

Mirror run_v2_sweep.py structure but specifically for GPT4TS backbone:
6 dataset_types × 3 seeds × (pretrain + c4_repr + 3 × c5_head) = 90 units.

Reuses seedcells_v2.json dataset_types definition.
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
SEEDCELLS = REPO_ROOT / "paper_supplement" / "protocol" / "classification_seedcells_v2.json"
OUT_BASE = REPO_ROOT / "SimMTM_Classification" / "outputs" / "v3_gpt4ts_sweep"
WRAPPER = REPO_ROOT / "paper_supplement" / "scripts" / "gpt4ts_wrapper.py"


def now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_done(tsv: Path) -> set:
    if not tsv.exists():
        return set()
    done = set()
    with open(tsv) as f:
        next(f, None)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[1] == "DONE":
                done.add(parts[0])
    return done


def append_status(tsv: Path, cell_id: str, status: str):
    new = not tsv.exists()
    with open(tsv, "a") as f:
        if new:
            f.write("cell_id\tstatus\ttimestamp\n")
        f.write(f"{cell_id}\t{status}\t{now_iso()}\n")


def run_cmd(cmd, log_path: Path, gpu: str):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "TMPDIR": "/workspace/tmp"}
    with open(log_path, "ab") as f:
        f.write(f"\n=== {now_iso()} CMD: {' '.join(map(str, cmd))} ===\n".encode())
        f.flush()
        proc = subprocess.run([str(c) for c in cmd], stdout=f,
                              stderr=subprocess.STDOUT, env=env)
    return proc.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--out_dir", default=str(OUT_BASE))
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    out_base = Path(args.out_dir); out_base.mkdir(parents=True, exist_ok=True)
    status_tsv = out_base / "run_status.tsv"
    done = read_done(status_tsv)

    spec = json.loads(SEEDCELLS.read_text())
    dataset_types = spec["dataset_types"]
    seeds = spec["config"]["seeds"]
    losses = spec["config"]["losses_c5"]

    backbone = "GPT4TS"
    total = 0
    for dt in dataset_types:
        for seed in seeds:
            cell_dir = out_base / dt["id"] / f"seed{seed}"
            cell_dir.mkdir(parents=True, exist_ok=True)

            # Pretrain
            pre_id = f"{backbone}|{dt['id']}|seed{seed}|pretrain"
            total += 1
            if pre_id in done:
                print(f"[skip] {pre_id}", flush=True)
            else:
                print(f"\n[{now_iso()}] === {pre_id} ===", flush=True)
                cmd = ["/usr/bin/python3", "-u", str(WRAPPER), "pretrain",
                       "--pretrain_dataset", dt["source"],
                       "--seed", str(seed), "--gpu", args.gpu,
                       "--out_dir", str(cell_dir),
                       "--epochs_pretrain", "10"]
                if args.dry_run:
                    append_status(status_tsv, pre_id, "DONE_DRY")
                else:
                    rc = run_cmd(cmd, cell_dir / "pretrain.log", args.gpu)
                    ok = rc == 0 and (cell_dir / "encoder.pt").exists()
                    append_status(status_tsv, pre_id, "DONE" if ok else "FAIL")
                    if not ok:
                        print(f"  pretrain FAIL — skipping cell", flush=True)
                        continue

            # C-4 repr
            c4_id = f"{backbone}|{dt['id']}|seed{seed}|c4"
            total += 1
            if c4_id in done:
                print(f"[skip] {c4_id}", flush=True)
            else:
                print(f"[{now_iso()}] {c4_id}", flush=True)
                cmd = ["/usr/bin/python3", "-u", str(WRAPPER), "c4_repr",
                       "--target_dataset", dt["target"],
                       "--encoder_path", str(cell_dir / "encoder.pt"),
                       "--seed", str(seed), "--gpu", args.gpu,
                       "--epochs_decoder", "50",
                       "--out_json", str(cell_dir / "c4_result.json")]
                if args.dry_run:
                    append_status(status_tsv, c4_id, "DONE_DRY")
                else:
                    rc = run_cmd(cmd, cell_dir / "c4.log", args.gpu)
                    append_status(status_tsv, c4_id, "DONE" if rc == 0 else "FAIL")

            # C-5 heads (3 losses)
            for loss in losses:
                c5_id = f"{backbone}|{dt['id']}|seed{seed}|c5_{loss}"
                total += 1
                if c5_id in done:
                    print(f"[skip] {c5_id}", flush=True)
                    continue
                print(f"[{now_iso()}] {c5_id}", flush=True)
                cmd = ["/usr/bin/python3", "-u", str(WRAPPER), "c5_head",
                       "--target_dataset", dt["target"],
                       "--encoder_path", str(cell_dir / "encoder.pt"),
                       "--recon_loss", loss,
                       "--seed", str(seed), "--gpu", args.gpu,
                       "--epochs_head", "30",
                       "--epochs_classifier", "50",
                       "--out_json", str(cell_dir / f"c5_{loss}_result.json")]
                if args.dry_run:
                    append_status(status_tsv, c5_id, "DONE_DRY")
                else:
                    rc = run_cmd(cmd, cell_dir / f"c5_{loss}.log", args.gpu)
                    append_status(status_tsv, c5_id, "DONE" if rc == 0 else "FAIL")

    print(f"\n[{now_iso()}] v3 GPT4TS sweep complete. Total units: {total}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
