#!/usr/bin/env python3
"""GPT4TS backbone wrapper for AAAI27 protocol v2 Section 17 (modern LLM backbone).

Mirrors tfc/ts2vec wrapper pattern: pretrain | c5_head | c4_repr.

GPT4TS uses HuggingFace pretrained GPT-2 (frozen except LayerNorm + position
embedding) as a time-series encoder via patch embedding. NO additional
pretraining of the GPT-2 weights — we use the official frozen-LM approach.

For our framework:
  - pretrain mode: actually a quick "warmup" of the patch embedding + LN +
    classification head on the source dataset (standard GPT4TS training).
    Saves encoder snapshot.
  - c5_head: load encoder, freeze, attach our recon head trained with chosen
    loss, finetune classifier.
  - c4_repr: load encoder, freeze, linear decoder, measure 8 metrics.

Reuses our .pt data format (B, C, T) — converts to (B, T, C) for GPT4TS.
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
GPT4TS_ROOT = REPO_ROOT / "backbones" / "GPT4TS" / "Classification"
SIMMTM_FORECAST = REPO_ROOT / "SimMTM_Forecasting"

# Path: import order matters for module shadowing
sys.path.insert(0, str(GPT4TS_ROOT / "src"))


def _load_by_file(name: str, rel_path: str):
    """Load SimMTM_Forecasting modules via file path (avoid TS2Vec/GPT4TS shadow)."""
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


def _pcc(p, t, eps=1e-8):
    import torch.nn.functional as F
    p_c = p - p.mean(dim=-1, keepdim=True)
    t_c = t - t.mean(dim=-1, keepdim=True)
    return torch.mean(F.cosine_similarity(p_c, t_c, dim=-1, eps=eps))


def _si_snr(p, t, eps=1e-8):
    p = p - p.mean(dim=-1, keepdim=True)
    t = t - t.mean(dim=-1, keepdim=True)
    s = torch.sum(p * t, dim=-1, keepdim=True) * t / (torch.sum(t * t, dim=-1, keepdim=True) + eps)
    e = p - s
    snr = 10 * torch.log10(torch.sum(s * s, dim=-1) / (torch.sum(e * e, dim=-1) + eps) + eps)
    return torch.mean(snr)


# ── data loading (same format as other wrappers) ────────────────────────
def _load_dataset_pt(name: str, split: str):
    path = REPO_ROOT / "backbones" / "TFC" / "datasets" / name / f"{split}.pt"
    d = torch.load(path, weights_only=False, map_location="cpu")
    x = d["samples"].float()
    y = d["labels"].long()
    if x.ndim == 2:
        x = x.unsqueeze(1)
    return x, y


class _MockData:
    """Minimal interface GPT4TS's gpt4ts model expects (max_seq_len, feature_df.shape[1], class_names)."""
    def __init__(self, x, y, channel=1):
        import pandas as pd
        self.max_seq_len = x.shape[-1]
        self.feature_df = pd.DataFrame(np.zeros((1, channel)))
        self.class_names = sorted(set(y.tolist()))


# ── GPT4TS encoder forward ──────────────────────────────────────────────
def _build_gpt4ts(x_train, y_train, device, d_model=768, patch_size=8, stride=4, dropout=0.1):
    """Instantiate gpt4ts model. Returns (model, repr_dim).
    repr_dim = d_model * patch_num (post-flatten before out_layer)."""
    from models.gpt4ts import gpt4ts
    data = _MockData(x_train, y_train, channel=1)
    cfg = {
        "patch_size": patch_size,
        "stride": stride,
        "d_model": d_model,
        "dropout": dropout,
    }
    model = gpt4ts(cfg, data).to(device)
    patch_num = (data.max_seq_len - patch_size) // stride + 2  # +1 padding +1 unfold
    repr_dim = d_model * patch_num
    return model, repr_dim, patch_num


def _encode_gpt4ts(model, x_ctc):
    """Run gpt4ts encoder up to pre-classification representation.
    Returns (B, repr_dim) flattened representation.

    Internal: patches → embedding → GPT-2 → flatten. Bypass out_layer.
    """
    from einops import rearrange
    x = x_ctc[:, :1, :]  # take channel 0 to match TF-C/TS2Vec convention
    # gpt4ts.forward expects (B, L, M)
    x_blm = x.permute(0, 2, 1)
    B, L, M = x_blm.shape
    input_x = rearrange(x_blm, 'b l m -> b m l')
    input_x = model.padding_patch_layer(input_x)
    input_x = input_x.unfold(dimension=-1, size=model.patch_size, step=model.stride)
    input_x = rearrange(input_x, 'b m n p -> b n (p m)')
    enc_out = model.enc_embedding(input_x, None)
    outputs = model.gpt2(inputs_embeds=enc_out).last_hidden_state
    outputs = model.act(outputs)
    outputs = model.ln_proj(outputs.reshape(B, -1))
    return outputs  # (B, repr_dim)


# ── modes ────────────────────────────────────────────────────────────────
def cmd_pretrain(args):
    """Brief warmup of GPT4TS on source dataset, save encoder state."""
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    x_train, y_train = _load_dataset_pt(args.pretrain_dataset, "train")
    model, repr_dim, _ = _build_gpt4ts(x_train, y_train, device)

    x_train = x_train[:, :1, :].to(device); y_train = y_train.to(device)
    bs = min(16, len(x_train))
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    ce = nn.CrossEntropyLoss()

    n_train = len(x_train)
    for epoch in range(args.epochs_pretrain):
        model.train()
        perm = torch.randperm(n_train, device=device)
        ep = 0.0; nb = 0
        for i in range(0, n_train, bs):
            idx = perm[i:i+bs]
            x_blm = x_train[idx].permute(0, 2, 1)
            logits = model(x_blm, None)
            loss = ce(logits, y_train[idx])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            ep += loss.item(); nb += 1
        if epoch % 5 == 0:
            print(f"[gpt4ts pretrain epoch {epoch}] loss={ep/max(nb,1):.4f}", flush=True)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    enc_state = {
        "model_state_dict": model.state_dict(),
        "config": {"d_model": 768, "patch_size": 8, "stride": 4, "dropout": 0.1,
                   "n_train_seq_len": x_train.shape[-1], "n_classes": len(model.class_names) if hasattr(model, 'class_names') else int(y_train.max().item())+1},
    }
    out = out_dir / "encoder.pt"
    torch.save(enc_state, out)
    print(f"[gpt4ts pretrain] encoder saved -> {out}")
    return 0


def _load_encoder(args, device, x_train, y_train):
    ckpt = torch.load(args.encoder_path, weights_only=False, map_location=device)
    cfg = ckpt["config"]
    model, repr_dim, _ = _build_gpt4ts(x_train, y_train, device,
                                       d_model=cfg["d_model"],
                                       patch_size=cfg["patch_size"],
                                       stride=cfg["stride"],
                                       dropout=cfg["dropout"])
    try:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    except Exception as e:
        print(f"[gpt4ts] partial load (size mismatch expected for cross-domain): {e}")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, repr_dim


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


def _select_recon_loss(name):
    if name == "mse":  return nn.MSELoss()
    if name == "sdsc": return SignalDiceLossCanonical(alpha=10.0)
    if name == "zcr":  return DiffZCRLoss(alpha=10.0)
    raise ValueError(name)


def cmd_c5_head(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    x_train, y_train = _load_dataset_pt(args.target_dataset, "train")
    x_test,  y_test  = _load_dataset_pt(args.target_dataset, "test")
    model, repr_dim = _load_encoder(args, device, x_train, y_train)
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

    bs = min(8, len(x_train))
    n_train = len(x_train)
    for epoch in range(args.epochs_head):
        head.train()
        perm = torch.randperm(n_train, device=device)
        for i in range(0, n_train, bs):
            idx = perm[i:i+bs]
            z = _encode_gpt4ts(model, x_train[idx])
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
            z = _encode_gpt4ts(model, x_train[idx])
            logits = cls(z)
            loss = ce(logits, y_train[idx])
            optim_c.zero_grad(); loss.backward(); optim_c.step()

    cls.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for i in range(0, len(x_test), bs):
            z = _encode_gpt4ts(model, x_test[i:i+bs])
            preds = cls(z).argmax(dim=1)
            correct += (preds == y_test[i:i+bs]).sum().item()
            total += y_test[i:i+bs].numel()
    acc = correct / max(total, 1)
    out = {
        "backbone": "GPT4TS",
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


def cmd_c4_repr(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    x_train, y_train = _load_dataset_pt(args.target_dataset, "train")
    x_test,  _       = _load_dataset_pt(args.target_dataset, "test")
    model, repr_dim = _load_encoder(args, device, x_train, y_train)
    T = x_train.shape[2]
    x_train = x_train[:, :1, :].to(device)
    x_test  = x_test[:,  :1, :].to(device)

    decoder = nn.Linear(repr_dim, T).to(device)
    mse = nn.MSELoss()
    optim = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    bs = min(8, len(x_train))
    n_train = len(x_train)
    for epoch in range(args.epochs_decoder):
        decoder.train()
        perm = torch.randperm(n_train, device=device)
        for i in range(0, n_train, bs):
            idx = perm[i:i+bs]
            z = _encode_gpt4ts(model, x_train[idx])
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
            z = _encode_gpt4ts(model, x_test[i:i+bs])
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
        "PCC":   float(_pcc(pred, tgt).item()),
        "SI_SNR": float(_si_snr(pred, tgt).item()),
        "SDSC":  float(sdsc_metric(pred, tgt).item()),
    }
    out = {
        "backbone": "GPT4TS",
        "target": args.target_dataset,
        "seed": args.seed,
        "n_test": len(tgt),
        "metrics": metrics,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"[c4_repr] metrics: {metrics}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    p1 = sub.add_parser("pretrain")
    p1.add_argument("--pretrain_dataset", required=True)
    p1.add_argument("--seed", type=int, default=42)
    p1.add_argument("--gpu", default="0")
    p1.add_argument("--out_dir", required=True)
    p1.add_argument("--epochs_pretrain", type=int, default=10)

    p2 = sub.add_parser("c5_head")
    p2.add_argument("--target_dataset", required=True)
    p2.add_argument("--encoder_path", required=True)
    p2.add_argument("--recon_loss", choices=["mse", "sdsc", "zcr"], required=True)
    p2.add_argument("--seed", type=int, default=42)
    p2.add_argument("--gpu", default="0")
    p2.add_argument("--epochs_head", type=int, default=30)
    p2.add_argument("--epochs_classifier", type=int, default=50)
    p2.add_argument("--out_json", required=True)

    p3 = sub.add_parser("c4_repr")
    p3.add_argument("--target_dataset", required=True)
    p3.add_argument("--encoder_path", required=True)
    p3.add_argument("--seed", type=int, default=42)
    p3.add_argument("--gpu", default="0")
    p3.add_argument("--epochs_decoder", type=int, default=50)
    p3.add_argument("--out_json", required=True)

    args = ap.parse_args()
    if args.mode == "pretrain": return cmd_pretrain(args)
    if args.mode == "c5_head":  return cmd_c5_head(args)
    if args.mode == "c4_repr":  return cmd_c4_repr(args)


if __name__ == "__main__":
    raise SystemExit(main())
