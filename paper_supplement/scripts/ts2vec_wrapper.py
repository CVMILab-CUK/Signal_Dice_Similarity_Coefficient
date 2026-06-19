#!/usr/bin/env python3
"""TS2Vec backbone wrapper for AAAI27 protocol v2 cross-backbone generalization.

Same three modes as tfc_wrapper.py:

  --mode pretrain   : Train TS2Vec hierarchical-contrastive encoder. Saves to <out_dir>/encoder.pt.
  --mode c5_head    : Frozen encoder + recon head with --recon_loss + classifier finetune.
  --mode c4_repr    : Frozen encoder + linear decoder, measure 8 reconstruction metrics.

TS2Vec input convention is (B, T, C) — we transpose from our (B, C, T) .pt format.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
TS2VEC_ROOT = REPO_ROOT / "backbones" / "TS2Vec"
SIMMTM_FORECAST = REPO_ROOT / "SimMTM_Forecasting"

# CRITICAL: TS2Vec internally does `from models import TSEncoder` AND ships a
# top-level utils.py. Both shadow SimMTM_Forecasting's models/ and utils/
# packages, so we load SimMTM_Forecasting's modules via importlib file-path
# instead of sys.path.
sys.path.insert(0, str(TS2VEC_ROOT))
from ts2vec import TS2Vec
from models.encoder import TSEncoder


def _load_by_file(name: str, rel_path: str):
    """Load a SimMTM_Forecasting module via direct file path (avoids TS2Vec's
    top-level utils.py shadowing)."""
    import importlib.util
    full = SIMMTM_FORECAST / rel_path
    spec = importlib.util.spec_from_file_location(name, str(full))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_sdsc_mod = _load_by_file("sdsc_canonical_mod", "utils/sdsc_canonical.py")
SignalDiceCanonical = _sdsc_mod.SignalDiceCanonical
SignalDiceLossCanonical = _sdsc_mod.SignalDiceLossCanonical

_zcr_mod = _load_by_file("zcr_diff_mod", "utils/baselines/zcr_diff.py")
DiffZCRLoss = _zcr_mod.DiffZCRLoss
hard_zcr_metric = _zcr_mod.hard_zcr_metric

_quant_mod = _load_by_file("quantized_mse_mod", "utils/baselines/quantized_mse.py")
one_bit_mse = _quant_mod.one_bit_mse
two_bit_mu_law_mse = _quant_mod.two_bit_mu_law_mse


def _pearson_corr(p, t, eps=1e-8):
    """Inline PCC to avoid importing utils.metrics (which pulls TS2Vec utils.py).
    Mirrors SimMTM_Forecasting/utils/metrics.pearson_correlation."""
    import torch
    import torch.nn.functional as F
    p_c = p - p.mean(dim=-1, keepdim=True)
    t_c = t - t.mean(dim=-1, keepdim=True)
    return torch.mean(F.cosine_similarity(p_c, t_c, dim=-1, eps=eps))


def _si_snr(p, t, eps=1e-8):
    import torch
    p = p - p.mean(dim=-1, keepdim=True)
    t = t - t.mean(dim=-1, keepdim=True)
    s_tgt = torch.sum(p * t, dim=-1, keepdim=True) * t / (torch.sum(t * t, dim=-1, keepdim=True) + eps)
    e = p - s_tgt
    snr = 10 * torch.log10(torch.sum(s_tgt * s_tgt, dim=-1) / (torch.sum(e * e, dim=-1) + eps) + eps)
    return torch.mean(snr)


# ─── data loading (same format as TF-C wrapper) ──────────────────────────
def _load_dataset_pt(name: str, split: str):
    path = REPO_ROOT / "backbones" / "TFC" / "datasets" / name / f"{split}.pt"
    d = torch.load(path, weights_only=False, map_location="cpu")
    x = d["samples"].float()
    y = d["labels"].long()
    if x.ndim == 2:
        x = x.unsqueeze(1)
    return x, y


def _to_ts2vec_format(x: torch.Tensor) -> np.ndarray:
    """Our data is (B, C, T); TS2Vec wants (B, T, C). Single-channel projection
    for cross-backbone fairness (matches TF-C convention)."""
    x = x[:, :1, :]
    return x.permute(0, 2, 1).numpy()  # (B, T, 1)


# ─── pretrain mode ───────────────────────────────────────────────────────
def cmd_pretrain(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    x_train, _ = _load_dataset_pt(args.pretrain_dataset, "train")
    arr = _to_ts2vec_format(x_train)
    print(f"[ts2vec_wrapper] pretrain on {args.pretrain_dataset} shape={arr.shape}", flush=True)

    model = TS2Vec(
        input_dims=1,
        output_dims=args.output_dims,
        hidden_dims=args.hidden_dims,
        depth=args.depth,
        device=device,
        lr=args.lr,
        batch_size=args.batch_size,
        max_train_length=args.max_train_length,
    )
    loss_log = model.fit(arr, n_iters=args.n_iters, verbose=False)
    print(f"[ts2vec_wrapper] pretrain done. final_loss={loss_log[-1] if loss_log else 'NA'}")

    encoder_state = {
        "_net_state_dict": model._net.state_dict(),
        "config": {
            "input_dims": 1,
            "output_dims": args.output_dims,
            "hidden_dims": args.hidden_dims,
            "depth": args.depth,
        },
    }
    out_path = out_dir / "encoder.pt"
    torch.save(encoder_state, out_path)
    print(f"[ts2vec_wrapper] encoder saved -> {out_path}")
    return 0


def _build_encoder_from_ckpt(ckpt_path: str, device: str) -> TSEncoder:
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    cfg = ckpt["config"]
    enc = TSEncoder(
        input_dims=cfg["input_dims"],
        output_dims=cfg["output_dims"],
        hidden_dims=cfg["hidden_dims"],
        depth=cfg["depth"],
    ).to(device)
    enc.load_state_dict(ckpt["_net_state_dict"])
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False
    return enc, cfg["output_dims"]


def _encode(enc: TSEncoder, x_ctc: torch.Tensor) -> torch.Tensor:
    """Run TS2Vec encoder on (B, C, T) tensor; returns full-series max-pooled
    representation of shape (B, output_dims). TSEncoder returns (B, T, Co)
    (verified empirically). Pool over T (dim=1)."""
    x_bt = x_ctc.permute(0, 2, 1)  # (B, C, T) → (B, T, C)
    with torch.no_grad():
        out = enc(x_bt)            # (B, T, output_dims)
    return out.max(dim=1).values   # (B, output_dims)


# ─── c5_head ─────────────────────────────────────────────────────────────
class ReconHead(nn.Module):
    def __init__(self, repr_dim, T):
        super().__init__()
        self.fc = nn.Linear(repr_dim, T)
    def forward(self, z):
        return self.fc(z)


class Classifier(nn.Module):
    def __init__(self, repr_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(repr_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)
    def forward(self, z):
        return self.fc2(torch.sigmoid(self.fc1(z)))


def _select_recon_loss(name: str):
    if name == "mse":
        return nn.MSELoss()
    if name == "sdsc":
        return SignalDiceLossCanonical(alpha=10.0)
    if name == "zcr":
        return DiffZCRLoss(alpha=10.0)
    raise ValueError(name)


def cmd_c5_head(args):
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    enc, repr_dim = _build_encoder_from_ckpt(args.encoder_path, device)
    x_train, y_train = _load_dataset_pt(args.target_dataset, "train")
    x_test,  y_test  = _load_dataset_pt(args.target_dataset, "test")
    T = x_train.shape[2]
    x_train = x_train[:, :1, :].to(device); y_train = y_train.to(device)
    x_test  = x_test[:,  :1, :].to(device); y_test  = y_test.to(device)
    num_classes = int(y_train.max().item()) + 1

    head = ReconHead(repr_dim, T).to(device)
    cls  = Classifier(repr_dim, num_classes).to(device)
    recon_criterion = _select_recon_loss(args.recon_loss).to(device)
    ce = nn.CrossEntropyLoss()
    optim_h = torch.optim.Adam(head.parameters(), lr=1e-3)
    optim_c = torch.optim.Adam(cls.parameters(), lr=1e-3)

    bs = min(64, len(x_train))
    n_train = len(x_train)
    for epoch in range(args.epochs_head):
        head.train()
        perm = torch.randperm(n_train, device=device)
        for i in range(0, n_train, bs):
            idx = perm[i:i+bs]
            z = _encode(enc, x_train[idx])
            x_hat = head(z)
            tgt = x_train[idx].squeeze(1)
            loss = recon_criterion(x_hat, tgt)
            optim_h.zero_grad(); loss.backward(); optim_h.step()
        if epoch % 10 == 0:
            print(f"[c5_head epoch {epoch}] recon_loss={loss.item():.4f}", flush=True)

    for epoch in range(args.epochs_classifier):
        cls.train()
        perm = torch.randperm(n_train, device=device)
        for i in range(0, n_train, bs):
            idx = perm[i:i+bs]
            z = _encode(enc, x_train[idx])
            logits = cls(z)
            loss = ce(logits, y_train[idx])
            optim_c.zero_grad(); loss.backward(); optim_c.step()
        if epoch % 20 == 0:
            print(f"[c5_head classify epoch {epoch}] ce_loss={loss.item():.4f}", flush=True)

    cls.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for i in range(0, len(x_test), bs):
            z = _encode(enc, x_test[i:i+bs])
            preds = cls(z).argmax(dim=1)
            correct += (preds == y_test[i:i+bs]).sum().item()
            total += y_test[i:i+bs].numel()
    acc = correct / max(total, 1)
    out = {
        "backbone": "TS2Vec",
        "target": args.target_dataset,
        "recon_loss": args.recon_loss,
        "seed": args.seed,
        "accuracy": acc,
        "n_test": total,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"[c5_head] test acc={acc:.4f} -> {args.out_json}")
    return 0


# ─── c4_repr ─────────────────────────────────────────────────────────────
def cmd_c4_repr(args):
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    enc, repr_dim = _build_encoder_from_ckpt(args.encoder_path, device)
    x_train, _ = _load_dataset_pt(args.target_dataset, "train")
    x_test,  _ = _load_dataset_pt(args.target_dataset, "test")
    T = x_train.shape[2]
    x_train = x_train[:, :1, :].to(device)
    x_test  = x_test[:,  :1, :].to(device)

    decoder = nn.Linear(repr_dim, T).to(device)
    mse = nn.MSELoss()
    optim = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    bs = 64
    n_train = len(x_train)
    for epoch in range(args.epochs_decoder):
        decoder.train()
        perm = torch.randperm(n_train, device=device)
        for i in range(0, n_train, bs):
            idx = perm[i:i+bs]
            z = _encode(enc, x_train[idx])
            x_hat = decoder(z)
            tgt = x_train[idx].squeeze(1)
            loss = mse(x_hat, tgt)
            optim.zero_grad(); loss.backward(); optim.step()
        if epoch % 20 == 0:
            print(f"[c4_repr decoder epoch {epoch}] mse={loss.item():.4f}", flush=True)

    decoder.eval()
    recons = []; targets = []
    with torch.no_grad():
        for i in range(0, len(x_test), bs):
            z = _encode(enc, x_test[i:i+bs])
            recons.append(decoder(z))
            targets.append(x_test[i:i+bs].squeeze(1))
    pred = torch.cat(recons, dim=0)
    tgt  = torch.cat(targets, dim=0)

    sdsc_metric = SignalDiceCanonical()
    diff_zcr = DiffZCRLoss(alpha=10.0)
    metrics = {
        "MSE":   float(torch.mean((pred - tgt) ** 2).item()),
        "MAE":   float(torch.mean(torch.abs(pred - tgt)).item()),
        "ZCR_soft": float(diff_zcr(pred, tgt).item()),
        "ZCR_hard": float(hard_zcr_metric(pred, tgt).item()),
        "1bit":  float(one_bit_mse(pred, tgt).item()),
        "2bit":  float(two_bit_mu_law_mse(pred, tgt).item()),
        "PCC":   float(_pearson_corr(pred, tgt).item()),
        "SI_SNR": float(_si_snr(pred, tgt).item()),
        "SDSC":  float(sdsc_metric(pred, tgt).item()),
    }
    out = {
        "backbone": "TS2Vec",
        "target": args.target_dataset,
        "seed": args.seed,
        "n_test": len(tgt),
        "metrics": metrics,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"[c4_repr] metrics: {metrics}")
    return 0


# ─── entry ───────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    p1 = sub.add_parser("pretrain")
    p1.add_argument("--pretrain_dataset", required=True)
    p1.add_argument("--seed", type=int, default=42)
    p1.add_argument("--gpu", default="0")
    p1.add_argument("--out_dir", required=True)
    p1.add_argument("--n_iters", type=int, default=200)
    p1.add_argument("--output_dims", type=int, default=320)
    p1.add_argument("--hidden_dims", type=int, default=64)
    p1.add_argument("--depth", type=int, default=10)
    p1.add_argument("--lr", type=float, default=0.001)
    p1.add_argument("--batch_size", type=int, default=16)
    p1.add_argument("--max_train_length", type=int, default=3000)

    p2 = sub.add_parser("c5_head")
    p2.add_argument("--target_dataset", required=True)
    p2.add_argument("--encoder_path", required=True)
    p2.add_argument("--recon_loss", choices=["mse", "sdsc", "zcr"], required=True)
    p2.add_argument("--seed", type=int, default=42)
    p2.add_argument("--gpu", default="0")
    p2.add_argument("--epochs_head", type=int, default=50)
    p2.add_argument("--epochs_classifier", type=int, default=100)
    p2.add_argument("--out_json", required=True)

    p3 = sub.add_parser("c4_repr")
    p3.add_argument("--target_dataset", required=True)
    p3.add_argument("--encoder_path", required=True)
    p3.add_argument("--seed", type=int, default=42)
    p3.add_argument("--gpu", default="0")
    p3.add_argument("--epochs_decoder", type=int, default=80)
    p3.add_argument("--out_json", required=True)

    args = ap.parse_args()
    if args.mode == "pretrain":
        return cmd_pretrain(args)
    if args.mode == "c5_head":
        return cmd_c5_head(args)
    if args.mode == "c4_repr":
        return cmd_c4_repr(args)


if __name__ == "__main__":
    raise SystemExit(main())
