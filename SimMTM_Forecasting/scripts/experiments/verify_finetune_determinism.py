#!/usr/bin/env python3
"""Diagnose the identical-finetune-result bug.

Background: in an earlier multi_sweep_v1 run we observed that finetuning
SimMTM/ETTh1 starting from two distinct pretrain ckpts (mse vs sdsc loss
modes — different MD5) produced finetune Train/Val/Test losses that matched
to 7 decimal places across the entire epoch. This script attempts to
reproduce and isolate the cause without re-running the whole sweep.

Procedure:
    1. Load each candidate pretrain ckpt into its own model instance.
    2. Confirm encoder params differ.
    3. Run a single deterministic finetune mini-batch from each:
         a. Identical input batch (same seed, same indices).
         b. Compare forward outputs (must differ if encoders differ).
         c. Compare loss values (must differ).
    4. Optionally: run N steps of finetune training and report whether
       the two models converge to the same parameters.

Run only when the main sweep has produced fresh ckpts at
outputs/pretrain_checkpoints/ETTh1/{mse,sdsc}/ckpt_best.pth. Will not block
or affect the running sweep (runs on CPU by default to avoid GPU contention).

Usage:
    /usr/bin/python3 scripts/experiments/verify_finetune_determinism.py [--gpu]
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))


def build_args(loss_mode):
    """Replicate the argparse namespace the model __init__ expects."""
    class A: pass
    a = A()
    for k, v in dict(
        task_name='finetune', model='SimMTM', data='ETTh1', features='M',
        seq_len=96, label_len=48, pred_len=96, e_layers=3, d_layers=1,
        enc_in=7, dec_in=7, c_out=7, d_model=32, d_ff=64, n_heads=16,
        dropout=0.2, fc_dropout=0, head_dropout=0.1, embed='timeF',
        activation='gelu', output_attention=False, factor=1, distil=True,
        freq='h', positive_nums=3, mask_rate=0.5, lm=3, rbtp=1,
        temperature=0.2, masked_rule='geometric', individual=0,
        moving_avg=25, pct_start=0.3, top_k=5, num_kernels=3,
        patch_len=12, stride=12, alpha=None,
    ).items():
        setattr(a, k, v)
    a.loss_mode = loss_mode
    return a


def encoder_params_diff(m_a, m_b, prefix=("encoder", "enc_embedding")):
    """Return summary of how many encoder params differ between two models."""
    sd_a = m_a.state_dict(); sd_b = m_b.state_dict()
    n_same = n_diff = 0
    max_delta = 0.0
    for k in sd_a:
        if not any(k.startswith(p) for p in prefix):
            continue
        if k not in sd_b:
            continue
        a, b = sd_a[k].float(), sd_b[k].float()
        if a.shape != b.shape:
            continue
        if torch.equal(sd_a[k], sd_b[k]):
            n_same += 1
        else:
            n_diff += 1
            max_delta = max(max_delta, (a - b).abs().max().item())
    return n_same, n_diff, max_delta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-mse", default=str(REPO_ROOT / "outputs/pretrain_checkpoints/ETTh1/mse/ckpt_best.pth"))
    ap.add_argument("--ckpt-sdsc", default=str(REPO_ROOT / "outputs/pretrain_checkpoints/ETTh1/sdsc/ckpt_best.pth"))
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--seed", type=int, default=2023)
    ap.add_argument("--steps", type=int, default=5, help="Finetune steps to compare model trajectories")
    args = ap.parse_args()

    device = torch.device("cuda:0" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    for p in [args.ckpt_mse, args.ckpt_sdsc]:
        if not Path(p).exists():
            print(f"MISSING ckpt: {p}")
            sys.exit(1)

    # Reset RNG identically before each model build
    from models.SimMTM import Model
    from utils.tools import transfer_weights

    def fresh_model(ckpt_path, loss_mode):
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(args.seed)
        m = Model(build_args(loss_mode)).float()
        m = transfer_weights(ckpt_path, m, device='cpu')
        return m.to(device).train()

    m_mse = fresh_model(args.ckpt_mse, "mse")
    m_sdsc = fresh_model(args.ckpt_sdsc, "sdsc")

    # --- 1. Encoder param diff
    n_same, n_diff, max_d = encoder_params_diff(m_mse, m_sdsc)
    print(f"\n[1] Encoder params: same={n_same}, diff={n_diff}, max|delta|={max_d:.6e}")
    if n_diff == 0:
        print("    WARNING: encoders are identical. Bug may be at ckpt-save level.")
    else:
        print("    OK: encoders differ as expected.")

    # --- 2. Single forward pass with identical input
    torch.manual_seed(args.seed)
    bs, seq_len, n_vars = 16, 96, 7
    x = torch.randn(bs, seq_len, n_vars, device=device)
    x_mark = torch.zeros(bs, seq_len, 4, device=device)
    target = torch.randn(bs, 96, n_vars, device=device)

    y_mse = m_mse(x, x_mark)
    y_sdsc = m_sdsc(x, x_mark)
    out_delta = (y_mse - y_sdsc).abs().max().item()
    out_norm_mse = y_mse.norm().item()
    out_norm_sdsc = y_sdsc.norm().item()
    print(f"\n[2] Forward pass: out_norm mse={out_norm_mse:.4f}, sdsc={out_norm_sdsc:.4f}, max|delta|={out_delta:.6e}")
    if out_delta < 1e-6:
        print("    BUG CONFIRMED: forward outputs identical despite different encoders.")
    else:
        print("    OK: forward outputs differ as expected.")

    # --- 3. Single MSE loss
    loss_mse_to_target = torch.nn.functional.mse_loss(y_mse, target).item()
    loss_sdsc_to_target = torch.nn.functional.mse_loss(y_sdsc, target).item()
    print(f"\n[3] MSE(output, target): mse-ckpt={loss_mse_to_target:.7f}, sdsc-ckpt={loss_sdsc_to_target:.7f}")
    print(f"    diff = {abs(loss_mse_to_target - loss_sdsc_to_target):.6e}")

    # --- 4. N-step training trajectory comparison
    print(f"\n[4] {args.steps}-step finetune (Adam lr=1e-4, OneCycleLR pct_start=0.3)")
    from torch.optim.lr_scheduler import OneCycleLR
    def make_trainer(model):
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        # Mimic the production finetune scheduler
        sched = OneCycleLR(opt, max_lr=1e-4, total_steps=args.steps, pct_start=0.3)
        return opt, sched

    opt_mse, sch_mse = make_trainer(m_mse)
    opt_sdsc, sch_sdsc = make_trainer(m_sdsc)
    crit = torch.nn.MSELoss()

    for step in range(args.steps):
        torch.manual_seed(args.seed + step)
        x_step = torch.randn(bs, seq_len, n_vars, device=device)
        x_mark_step = torch.zeros(bs, seq_len, 4, device=device)
        target_step = torch.randn(bs, 96, n_vars, device=device)

        opt_mse.zero_grad()
        opt_sdsc.zero_grad()
        y1 = m_mse(x_step, x_mark_step)
        y2 = m_sdsc(x_step, x_mark_step)
        l1 = crit(y1, target_step)
        l2 = crit(y2, target_step)
        l1.backward(); opt_mse.step(); sch_mse.step()
        l2.backward(); opt_sdsc.step(); sch_sdsc.step()
        print(f"   step {step}: loss mse-ckpt={l1.item():.7f}, sdsc-ckpt={l2.item():.7f}, diff={abs(l1.item()-l2.item()):.6e}")

    # --- 5. Post-training encoder diff
    n_same, n_diff, max_d = encoder_params_diff(m_mse, m_sdsc)
    print(f"\n[5] After {args.steps} steps: encoder params same={n_same}, diff={n_diff}, max|delta|={max_d:.6e}")

    print("\nVerdict:")
    if out_delta < 1e-6:
        print("  ❌ Bug reproduced — forward outputs collapse to identical despite different encoders.")
        print("     Next steps: inspect `transfer_weights` modify-in-place semantics, check for")
        print("     downstream module that resets encoder (e.g., re-init in finetune branch).")
    elif abs(loss_mse_to_target - loss_sdsc_to_target) < 1e-7:
        print("  ❌ Bug reproduced at loss level — outputs differ but loss is identical (suspicious).")
    else:
        print("  ✅ No bug at single-step level. Finetune-level identicality likely from OneCycleLR")
        print("     LR-too-small at start. Try raising lr (e.g., 1e-3) or extending the smoke to")
        print("     a full epoch (528 steps).")


if __name__ == "__main__":
    main()
