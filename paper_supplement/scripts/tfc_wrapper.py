#!/usr/bin/env python3
"""TF-C backbone wrapper for AAAI27 protocol v2 cross-backbone generalization.

Three operation modes:

  --mode pretrain
      Pretrain TF-C with its original NT-Xent + time-freq consistency loss.
      Saves encoder + projector state to <out_dir>/encoder.pt.
      Reuses upstream TF-C main.py via subprocess (zero modifications to
      upstream code beyond config_files/{Epilepsy,Gesture,HAR}_Configs.py
      fixes for TSlength_aligned).

  --mode c5_head
      Load pretrained encoder from <encoder_path>, freeze it, attach a
      post-hoc reconstruction head, train the head with --recon_loss in
      {mse, sdsc, zcr}, then attach a classifier and finetune both.
      Writes JSON {accuracy, recon_loss_final, classify_loss_final}.

  --mode c4_repr
      Load pretrained encoder, freeze, fit a linear decoder z -> x_hat,
      measure 8 metrics on x_hat vs x (test set). Writes JSON with per-
      metric scores.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import torch.fft as fft
import torch.nn as nn
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
TFC_CODE = REPO_ROOT / "backbones" / "TFC" / "code"
TFC_DATA = REPO_ROOT / "backbones" / "TFC" / "datasets"

# Make TF-C upstream code importable
sys.path.insert(0, str(TFC_CODE))
sys.path.insert(0, str(TFC_CODE / "TFC"))

# Make our canonical SDSC + baselines importable
sys.path.insert(0, str(REPO_ROOT / "SimMTM_Forecasting"))


def _load_dataset_pt(name: str, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Load (samples, labels) from our /workspace .pt files. TF-C convention:
    take only the first channel and clip to TSlength_aligned (done at call site).
    """
    path = REPO_ROOT / "backbones" / "TFC" / "datasets" / name / f"{split}.pt"
    d = torch.load(path, weights_only=False, map_location="cpu")
    x = d["samples"].float()
    y = d["labels"].long()
    # Ensure (N, C, T)
    if x.ndim == 2:
        x = x.unsqueeze(1)
    return x, y


def _import_tfc_config(name: str):
    from importlib import import_module
    return import_module(f"config_files.{name}_Configs").Config()


def _build_tfc_model(configs):
    """Instantiate TF-C model + projector. Restored from upstream model.py."""
    from TFC.model import TFC as TFCModel
    return TFCModel(configs)


def _to_tf_pair(x: torch.Tensor, T: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Project x to first channel, clip to T, build (x_time, x_freq) tuple."""
    x = x[:, :1, :T]
    if x.shape[2] < T:
        pad = torch.zeros(x.shape[0], 1, T - x.shape[2], dtype=x.dtype)
        x = torch.cat([x, pad], dim=2)
    x_t = x
    x_f = fft.fft(x).abs()
    return x_t, x_f


# ────────────────────────── pretrain mode ────────────────────────────────
def cmd_pretrain(args):
    """Subprocess TF-C upstream main.py to pretrain. Copies the resulting
    ckpt to <out_dir>/encoder.pt so c5_head / c4_repr can find it.
    """
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "tfc_pretrain.log"

    target = args.target_dataset or args.pretrain_dataset
    cmd = [
        "/usr/bin/python3", "-u", "main.py",
        "--seed", str(args.seed),
        "--training_mode", "pre_train",
        "--pretrain_dataset", args.pretrain_dataset,
        "--target_dataset", target,
        "--logs_save_dir", str(out_dir),
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu),
           "TMPDIR": "/workspace/tmp"}
    print(f"[tfc_wrapper] pretrain: {' '.join(cmd)}", flush=True)
    with open(log_path, "wb") as f:
        proc = subprocess.run(cmd, cwd=str(TFC_CODE / "TFC"), stdout=f,
                              stderr=subprocess.STDOUT, env=env)
    if proc.returncode != 0:
        print(f"[tfc_wrapper] pretrain FAILED rc={proc.returncode}")
        return proc.returncode

    # Locate the upstream saved checkpoint and copy to canonical location.
    pattern = (f"{args.pretrain_dataset}_2_{target}/run1/"
               f"pre_train_seed_{args.seed}_2layertransformer/saved_models/ckp_last.pt")
    src = out_dir / pattern
    if not src.exists():
        # Search recursively
        candidates = list(out_dir.rglob("ckp_last.pt"))
        if not candidates:
            print(f"[tfc_wrapper] no ckp_last.pt found in {out_dir}")
            return 1
        src = candidates[0]
    dst = out_dir / "encoder.pt"
    if not dst.exists():
        dst.write_bytes(src.read_bytes())
    print(f"[tfc_wrapper] encoder saved -> {dst}")
    return 0


# ────────────────────────── c5_head mode ─────────────────────────────────
class ReconHead(nn.Module):
    """Linear post-hoc reconstruction head: representation -> time-domain signal.

    Input: concat([z_time, z_freq]) of dimension 256.
    Output: T-dimensional time-domain reconstruction.
    """
    def __init__(self, repr_dim: int, T: int):
        super().__init__()
        self.fc = nn.Linear(repr_dim, T)

    def forward(self, z):
        return self.fc(z)


class Classifier(nn.Module):
    """Match TF-C upstream target_classifier."""
    def __init__(self, repr_dim: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(repr_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, z):
        h = torch.sigmoid(self.fc1(z))
        return self.fc2(h)


def _select_recon_loss(name: str):
    from utils.sdsc_canonical import SignalDiceLossCanonical
    from utils.baselines.zcr_diff import DiffZCRLoss
    if name == "mse":
        return nn.MSELoss()
    if name == "sdsc":
        return SignalDiceLossCanonical(alpha=10.0)
    if name == "zcr":
        return DiffZCRLoss(alpha=10.0)
    raise ValueError(f"unknown recon loss: {name}")


def _encoder_forward(model, x_t, x_f):
    """Returns concat([z_time, z_freq]) of shape (B, 256)."""
    _h_t, z_t, _h_f, z_f = model(x_t, x_f)
    return torch.cat([z_t, z_f], dim=1)


def cmd_c5_head(args):
    """Frozen TF-C encoder + recon head + classifier. AAAI27 protocol v2 AC-CL2-3/4."""
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    target = args.target_dataset
    configs = _import_tfc_config(target)
    T = configs.TSlength_aligned

    # Load data
    x_train, y_train = _load_dataset_pt(target, "train")
    x_val,   y_val   = _load_dataset_pt(target, "val")
    x_test,  y_test  = _load_dataset_pt(target, "test")
    x_train_t, x_train_f = _to_tf_pair(x_train, T)
    x_val_t,   x_val_f   = _to_tf_pair(x_val,   T)
    x_test_t,  x_test_f  = _to_tf_pair(x_test,  T)

    # Build model + load pretrained encoder
    model = _build_tfc_model(configs).to(device)
    ckpt = torch.load(args.encoder_path, weights_only=False, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    repr_dim = 256
    recon_head = ReconHead(repr_dim, T).to(device)
    classifier = Classifier(repr_dim, configs.num_classes_target).to(device)
    recon_criterion = _select_recon_loss(args.recon_loss).to(device)
    ce = nn.CrossEntropyLoss()

    optim_recon = torch.optim.Adam(recon_head.parameters(), lr=configs.lr)
    optim_cls   = torch.optim.Adam(classifier.parameters(), lr=configs.lr_f)

    # Move tensors to device
    x_train_t = x_train_t.to(device); x_train_f = x_train_f.to(device); y_train = y_train.to(device)
    x_test_t  = x_test_t.to(device);  x_test_f  = x_test_f.to(device);  y_test  = y_test.to(device)

    bs = min(configs.batch_size, len(x_train_t))
    n_epoch_head = args.epochs_head
    n_epoch_cls = args.epochs_classifier

    # Train recon head on train set with selected loss
    n_train = len(x_train_t)
    for epoch in range(n_epoch_head):
        recon_head.train()
        perm = torch.randperm(n_train, device=device)
        ep_loss = 0.0; nb = 0
        for i in range(0, n_train, bs):
            idx = perm[i:i+bs]
            with torch.no_grad():
                z = _encoder_forward(model, x_train_t[idx], x_train_f[idx])
            x_hat = recon_head(z)
            tgt = x_train_t[idx].squeeze(1)
            loss = recon_criterion(x_hat, tgt)
            optim_recon.zero_grad()
            loss.backward()
            optim_recon.step()
            ep_loss += loss.item(); nb += 1
        if epoch % 10 == 0:
            print(f"[c5_head epoch {epoch}] recon_loss={ep_loss/max(nb,1):.4f}", flush=True)

    # Train classifier on top
    for epoch in range(n_epoch_cls):
        classifier.train()
        perm = torch.randperm(n_train, device=device)
        ep_loss = 0.0; nb = 0
        for i in range(0, n_train, bs):
            idx = perm[i:i+bs]
            with torch.no_grad():
                z = _encoder_forward(model, x_train_t[idx], x_train_f[idx])
            logits = classifier(z)
            loss = ce(logits, y_train[idx])
            optim_cls.zero_grad()
            loss.backward()
            optim_cls.step()
            ep_loss += loss.item(); nb += 1
        if epoch % 20 == 0:
            print(f"[c5_head classify epoch {epoch}] ce_loss={ep_loss/max(nb,1):.4f}", flush=True)

    # Test accuracy
    classifier.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for i in range(0, len(x_test_t), bs):
            z = _encoder_forward(model, x_test_t[i:i+bs], x_test_f[i:i+bs])
            preds = classifier(z).argmax(dim=1)
            correct += (preds == y_test[i:i+bs]).sum().item()
            total += y_test[i:i+bs].numel()
    acc = correct / max(total, 1)
    out = {
        "backbone": "TFC",
        "target": target,
        "recon_loss": args.recon_loss,
        "seed": args.seed,
        "accuracy": acc,
        "n_test": total,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"[c5_head] test acc={acc:.4f}  -> {args.out_json}")
    return 0


# ────────────────────────── c4_repr mode ─────────────────────────────────
def cmd_c4_repr(args):
    """Fit a linear decoder z -> x_hat on train, measure 8 metrics on test."""
    from utils.sdsc_canonical import SignalDiceCanonical
    from utils.baselines.zcr_diff import DiffZCRLoss, hard_zcr_metric
    from utils.baselines.quantized_mse import one_bit_mse, two_bit_mu_law_mse
    from utils.metrics import pearson_correlation, si_snr

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    target = args.target_dataset
    configs = _import_tfc_config(target)
    T = configs.TSlength_aligned

    x_train, _ = _load_dataset_pt(target, "train")
    x_test,  _ = _load_dataset_pt(target, "test")
    x_train_t, x_train_f = _to_tf_pair(x_train, T)
    x_test_t,  x_test_f  = _to_tf_pair(x_test,  T)

    model = _build_tfc_model(configs).to(device)
    ckpt = torch.load(args.encoder_path, weights_only=False, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Linear decoder z -> x
    decoder = nn.Linear(256, T).to(device)
    optim = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    mse = nn.MSELoss()

    x_train_t = x_train_t.to(device); x_train_f = x_train_f.to(device)
    x_test_t  = x_test_t.to(device);  x_test_f  = x_test_f.to(device)
    bs = 64
    n_train = len(x_train_t)
    for epoch in range(args.epochs_decoder):
        decoder.train()
        perm = torch.randperm(n_train, device=device)
        ep = 0.0; nb = 0
        for i in range(0, n_train, bs):
            idx = perm[i:i+bs]
            with torch.no_grad():
                z = _encoder_forward(model, x_train_t[idx], x_train_f[idx])
            x_hat = decoder(z)
            tgt = x_train_t[idx].squeeze(1)
            loss = mse(x_hat, tgt)
            optim.zero_grad(); loss.backward(); optim.step()
            ep += loss.item(); nb += 1
        if epoch % 20 == 0:
            print(f"[c4_repr decoder epoch {epoch}] mse={ep/max(nb,1):.4f}", flush=True)

    # Test-set reconstructions
    decoder.eval()
    recons = []
    targets = []
    with torch.no_grad():
        for i in range(0, len(x_test_t), bs):
            z = _encoder_forward(model, x_test_t[i:i+bs], x_test_f[i:i+bs])
            recons.append(decoder(z))
            targets.append(x_test_t[i:i+bs].squeeze(1))
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
        "PCC":   float(pearson_correlation(pred, tgt).item()),
        "SI_SNR": float(si_snr(pred, tgt).item()),
        "SDSC":  float(sdsc_metric(pred, tgt).item()),
    }
    out = {
        "backbone": "TFC",
        "target": target,
        "seed": args.seed,
        "n_test": len(tgt),
        "metrics": metrics,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"[c4_repr] metrics: {metrics}")
    return 0


# ────────────────────────── entry ────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    ap_pre = sub.add_parser("pretrain")
    ap_pre.add_argument("--pretrain_dataset", required=True)
    ap_pre.add_argument("--target_dataset", default=None)
    ap_pre.add_argument("--seed", type=int, default=42)
    ap_pre.add_argument("--gpu", default="0")
    ap_pre.add_argument("--out_dir", required=True)

    ap_c5 = sub.add_parser("c5_head")
    ap_c5.add_argument("--target_dataset", required=True)
    ap_c5.add_argument("--encoder_path", required=True)
    ap_c5.add_argument("--recon_loss", choices=["mse", "sdsc", "zcr"], required=True)
    ap_c5.add_argument("--seed", type=int, default=42)
    ap_c5.add_argument("--gpu", default="0")
    ap_c5.add_argument("--epochs_head", type=int, default=50)
    ap_c5.add_argument("--epochs_classifier", type=int, default=100)
    ap_c5.add_argument("--out_json", required=True)

    ap_c4 = sub.add_parser("c4_repr")
    ap_c4.add_argument("--target_dataset", required=True)
    ap_c4.add_argument("--encoder_path", required=True)
    ap_c4.add_argument("--seed", type=int, default=42)
    ap_c4.add_argument("--gpu", default="0")
    ap_c4.add_argument("--epochs_decoder", type=int, default=80)
    ap_c4.add_argument("--out_json", required=True)

    args = ap.parse_args()
    if args.mode == "pretrain":
        return cmd_pretrain(args)
    if args.mode == "c5_head":
        return cmd_c5_head(args)
    if args.mode == "c4_repr":
        return cmd_c4_repr(args)


if __name__ == "__main__":
    raise SystemExit(main())
