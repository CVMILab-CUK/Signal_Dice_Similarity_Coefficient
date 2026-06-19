#!/usr/bin/env python3
"""AAAI27 protocol v2 cross-backbone sweep driver.

Iterates over (backbone, dataset_type, seed) → pretrains encoder, runs C-4
representation measurement, then for each loss runs C-5 head training.
Skips DONE cells via TSV resume.

Layout:
  outputs/v2_sweep/
    {backbone}/{dataset_type}/seed{seed}/encoder.pt
                                        /c4_result.json
                                        /c5_{loss}_result.json
    run_status.tsv  (cell_id, status, timestamp)
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
OUT_BASE = REPO_ROOT / "SimMTM_Classification" / "outputs" / "v2_sweep"
WRAPPERS = {
    "TFC": REPO_ROOT / "paper_supplement" / "scripts" / "tfc_wrapper.py",
    "TS2Vec": REPO_ROOT / "paper_supplement" / "scripts" / "ts2vec_wrapper.py",
}


def now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_done(tsv_path: Path) -> set:
    done = set()
    if not tsv_path.exists():
        return done
    with open(tsv_path) as f:
        next(f, None)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[1] == "DONE":
                done.add(parts[0])
    return done


def append_status(tsv_path: Path, cell_id: str, status: str):
    new = not tsv_path.exists()
    with open(tsv_path, "a") as f:
        if new:
            f.write("cell_id\tstatus\ttimestamp\n")
        f.write(f"{cell_id}\t{status}\t{now_iso()}\n")


def run_cmd(cmd, log_path: Path, env_extra=None) -> int:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(env_extra.get("gpu", "0") if env_extra else "0"),
           "TMPDIR": "/workspace/tmp"}
    with open(log_path, "ab") as f:
        f.write(f"\n=== {now_iso()} CMD: {' '.join(map(str, cmd))} ===\n".encode())
        f.flush()
        proc = subprocess.run([str(c) for c in cmd], stdout=f,
                              stderr=subprocess.STDOUT, env=env)
    return proc.returncode


def run_one_pretrain(backbone, dt, seed, out_dir, gpu):
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "pretrain.log"
    wrapper = WRAPPERS[backbone]
    if backbone == "TFC":
        cmd = ["/usr/bin/python3", "-u", str(wrapper), "pretrain",
               "--pretrain_dataset", dt["source"],
               "--target_dataset", dt["target"],
               "--seed", seed, "--gpu", gpu,
               "--out_dir", str(out_dir)]
    else:  # TS2Vec
        cmd = ["/usr/bin/python3", "-u", str(wrapper), "pretrain",
               "--pretrain_dataset", dt["source"],
               "--seed", seed, "--gpu", gpu,
               "--out_dir", str(out_dir),
               "--n_iters", "200"]
    rc = run_cmd(cmd, log, env_extra={"gpu": gpu})
    return rc == 0 and (out_dir / "encoder.pt").exists()


def run_c4(backbone, dt, seed, out_dir, gpu) -> bool:
    wrapper = WRAPPERS[backbone]
    encoder = out_dir / "encoder.pt"
    if not encoder.exists():
        return False
    json_out = out_dir / "c4_result.json"
    log = out_dir / "c4.log"
    cmd = ["/usr/bin/python3", "-u", str(wrapper), "c4_repr",
           "--target_dataset", dt["target"],
           "--encoder_path", str(encoder),
           "--seed", seed, "--gpu", gpu,
           "--out_json", str(json_out)]
    rc = run_cmd(cmd, log, env_extra={"gpu": gpu})
    return rc == 0 and json_out.exists()


def run_c5(backbone, dt, seed, loss, out_dir, gpu) -> bool:
    wrapper = WRAPPERS[backbone]
    encoder = out_dir / "encoder.pt"
    if not encoder.exists():
        return False
    json_out = out_dir / f"c5_{loss}_result.json"
    log = out_dir / f"c5_{loss}.log"
    cmd = ["/usr/bin/python3", "-u", str(wrapper), "c5_head",
           "--target_dataset", dt["target"],
           "--encoder_path", str(encoder),
           "--recon_loss", loss,
           "--seed", seed, "--gpu", gpu,
           "--out_json", str(json_out)]
    rc = run_cmd(cmd, log, env_extra={"gpu": gpu})
    return rc == 0 and json_out.exists()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--backbones", default="TFC,TS2Vec",
                    help="Comma-separated subset, default both.")
    ap.add_argument("--out_dir", default=str(OUT_BASE))
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    out_base = Path(args.out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    status_tsv = out_base / "run_status.tsv"
    done = read_done(status_tsv)

    spec = json.loads(SEEDCELLS.read_text())
    backbones = [b for b in args.backbones.split(",") if b in spec["config"]["backbones"]]
    dataset_types = spec["dataset_types"]
    seeds = spec["config"]["seeds"]
    losses = spec["config"]["losses_c5"]

    total_units = 0
    for backbone in backbones:
        for dt in dataset_types:
            for seed in seeds:
                cell_dir = out_base / backbone / dt["id"] / f"seed{seed}"
                # Pretrain unit
                pretrain_id = f"{backbone}|{dt['id']}|seed{seed}|pretrain"
                total_units += 1
                if pretrain_id in done:
                    print(f"[skip] {pretrain_id}", flush=True)
                else:
                    print(f"\n[{now_iso()}] === {pretrain_id} ===", flush=True)
                    if args.dry_run:
                        print(f"  (dry-run) cell_dir={cell_dir}")
                        append_status(status_tsv, pretrain_id, "DONE_DRY")
                    else:
                        ok = run_one_pretrain(backbone, dt, str(seed), cell_dir, args.gpu)
                        append_status(status_tsv, pretrain_id, "DONE" if ok else "FAIL")
                        if not ok:
                            print(f"  pretrain FAIL — skipping C-4/C-5 for this cell", flush=True)
                            continue

                # C-4 unit
                c4_id = f"{backbone}|{dt['id']}|seed{seed}|c4"
                total_units += 1
                if c4_id in done:
                    print(f"[skip] {c4_id}", flush=True)
                else:
                    print(f"[{now_iso()}] {c4_id}", flush=True)
                    if args.dry_run:
                        append_status(status_tsv, c4_id, "DONE_DRY")
                    else:
                        ok = run_c4(backbone, dt, str(seed), cell_dir, args.gpu)
                        append_status(status_tsv, c4_id, "DONE" if ok else "FAIL")

                # C-5 units (one per loss)
                for loss in losses:
                    c5_id = f"{backbone}|{dt['id']}|seed{seed}|c5_{loss}"
                    total_units += 1
                    if c5_id in done:
                        print(f"[skip] {c5_id}", flush=True)
                        continue
                    print(f"[{now_iso()}] {c5_id}", flush=True)
                    if args.dry_run:
                        append_status(status_tsv, c5_id, "DONE_DRY")
                    else:
                        ok = run_c5(backbone, dt, str(seed), loss, cell_dir, args.gpu)
                        append_status(status_tsv, c5_id, "DONE" if ok else "FAIL")

    print(f"\n[{now_iso()}] Sweep complete. Total units processed: {total_units}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
