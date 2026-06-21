"""DLinear (Zeng et al., AAAI 2023) non-Transformer baseline integrated into the
SDSC sweep harness. Same Model class interface as PatchTST/SimMTM so the existing
exp_simmtm.py finetune + pretrain pipeline can use it without modification.

Finetune (the AAAI27 use case): standard DLinear — series decomposition into
seasonal + trend, two channel-independent Linear maps from seq_len -> pred_len.

Pretrain: minimal masked-reconstruction wrapper using the same Linear layers
mapped seq_len -> seq_len. Returns the 15-tuple signature expected by
exp_simmtm.py so the sweep driver does not have to special-case DLinear.
loss_cl is set to zero (DLinear has no contrastive head); awl gates the target
loss alone.
"""

import torch
import torch.nn as nn
from utils.losses import (AutomaticWeightedLoss, SignalDiceLoss, mae_loss,
                          dtw_loss, ZCRLoss, DILATELoss)
from utils.metrics import SignalDice as SDSC, pearson_correlation, si_snr


class series_decomp(nn.Module):
    """Moving average based decomposition into (residual=seasonal, trend)."""

    def __init__(self, kernel_size):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        # x: [B, T, C]
        pad = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, pad, 1)
        end = x[:, -1:, :].repeat(1, pad, 1)
        x_pad = torch.cat([front, x, end], dim=1)
        trend = self.avg(x_pad.permute(0, 2, 1)).permute(0, 2, 1)
        seasonal = x - trend
        return seasonal, trend


class DLinearCore(nn.Module):
    """Two Linear maps (seasonal/trend) over time, channel-independent or shared."""

    def __init__(self, in_len, out_len, channels, individual=False):
        super().__init__()
        self.individual = individual
        self.channels = channels
        if individual:
            self.lin_s = nn.ModuleList([nn.Linear(in_len, out_len) for _ in range(channels)])
            self.lin_t = nn.ModuleList([nn.Linear(in_len, out_len) for _ in range(channels)])
        else:
            self.lin_s = nn.Linear(in_len, out_len)
            self.lin_t = nn.Linear(in_len, out_len)

    def forward(self, seasonal, trend):
        # inputs: [B, T_in, C]
        s = seasonal.permute(0, 2, 1)  # [B, C, T_in]
        t = trend.permute(0, 2, 1)
        if self.individual:
            outs = torch.stack([self.lin_s[i](s[:, i, :]) for i in range(self.channels)], dim=1)
            outt = torch.stack([self.lin_t[i](t[:, i, :]) for i in range(self.channels)], dim=1)
        else:
            outs = self.lin_s(s)
            outt = self.lin_t(t)
        return (outs + outt).permute(0, 2, 1)  # [B, T_out, C]


class Model(nn.Module):
    """DLinear with the SDSC sweep loss_mode interface.

    Same signature as PatchTST.Model / SimMTM.Model so exp_simmtm.py treats it
    interchangeably. Forecast path is standard DLinear. Pretrain path runs a
    very cheap masked reconstruction so the existing pretrain checkpoint
    pipeline produces a valid ckpt_best.pth for the finetune phase to load.
    """

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len
        self.configs = configs
        kernel = getattr(configs, "moving_avg", 25)
        individual = bool(getattr(configs, "individual", 0))
        channels = configs.enc_in

        self.decomp = series_decomp(kernel)

        if self.task_name == "finetune":
            self.encoder = DLinearCore(self.seq_len, self.pred_len, channels, individual)
        else:
            # pretrain: reconstruct masked input -> seq_len -> seq_len
            self.encoder = DLinearCore(self.seq_len, self.seq_len, channels, individual)

            self.mse = nn.MSELoss()
            self.sdsc = SignalDiceLoss(alpha=getattr(configs, "alpha", None))
            self.mae = mae_loss()
            self.pcc = pearson_correlation
            self.si_snr = si_snr
            self.dtw = dtw_loss(approx=True, use_cuda=False)
            self.zcr = ZCRLoss(alpha=10.0)
            self.dilate = DILATELoss(gamma_dilate=0.5, gamma_sdtw=0.01,
                                     use_cuda=torch.cuda.is_available())
            self.sdsc_metric = SDSC()

            if getattr(configs, "loss_mode", "hybrid") == "hybrid":
                self.awl = AutomaticWeightedLoss(2)  # rb + sd (no contrastive)
            else:
                self.awl = AutomaticWeightedLoss(1)  # target loss only

    def forecast(self, x_enc, x_mark_enc=None):
        # standard per-sample mean/var normalization a la Non-stationary TF
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_norm = x_enc / stdev

        seasonal, trend = self.decomp(x_norm)
        out = self.encoder(seasonal, trend)  # [B, pred_len, C]

        out = out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        out = out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return out

    def pretrain(self, x_enc, x_mark_enc, batch_x, mask):
        # masked reconstruction with shared DLinear core (seq_len -> seq_len)
        bs = batch_x.shape[0]
        means = (x_enc * mask).sum(1) / (mask.sum(1).clamp_min(1.0))
        means = means.unsqueeze(1).detach()
        x_c = (x_enc - means).masked_fill(mask == 0, 0.0)
        var = (x_c * x_c).sum(1) / (mask.sum(1).clamp_min(1.0))
        stdev = torch.sqrt(var + 1e-5).unsqueeze(1).detach()
        x_norm = x_c / stdev

        seasonal, trend = self.decomp(x_norm)
        rec = self.encoder(seasonal, trend)  # [B*(1+pos), seq_len, C]
        rec = rec * stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1)
        rec = rec + means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1)

        pred_batch_x = rec[:bs]
        target = batch_x.detach()

        metric_sd = self.sdsc_metric(pred_batch_x, target)
        metric_pcc = self.pcc(pred_batch_x, target)
        metric_si_snr = self.si_snr(pred_batch_x, target)

        device = pred_batch_x.device
        zero = torch.tensor(0., device=device)
        loss_cl = zero
        loss_rb = loss_sd = loss_mae = loss_dtw = loss_zcr = loss_dilate = zero

        lm = self.configs.loss_mode
        assert lm in {"mse", "sdsc", "mae", "dtw", "pcc", "snr", "zcr", "dilate", "hybrid"}, \
            f"DLinear: unsupported loss_mode {lm!r}"
        if lm == "mse":
            loss_rb = self.mse(pred_batch_x, target)
            loss = self.awl(loss_rb)
        elif lm == "sdsc":
            loss_sd = self.sdsc(pred_batch_x, target)
            loss = self.awl(loss_sd)
        elif lm == "mae":
            loss_mae = self.mae(pred_batch_x, target)
            loss = self.awl(loss_mae)
        elif lm == "dtw":
            loss_dtw = self.dtw(pred_batch_x, target)
            loss = self.awl(loss_dtw)
        elif lm == "pcc":
            loss = self.awl(1 - metric_pcc)
        elif lm == "snr":
            loss = self.awl(-metric_si_snr)
        elif lm == "zcr":
            loss_zcr = self.zcr(pred_batch_x, target)
            loss = self.awl(loss_zcr)
        elif lm == "dilate":
            loss_dilate = self.dilate(pred_batch_x, target)
            loss = self.awl(loss_dilate)
        else:  # hybrid
            loss_rb = self.mse(pred_batch_x, target)
            loss_sd = self.sdsc(pred_batch_x, target)
            loss = self.awl(loss_rb, loss_sd)

        # Tiny dummies so the exp_simmtm.py unpack + show_matrix() figure path do
        # not crash for DLinear (which has no contrastive head). Sized to a
        # constant 2x2 so Traffic (c_out=862) does not OOM on a real bs*c per
        # matrix — show_matrix only feeds these to seaborn.heatmap which works
        # on any shape.
        positives_mask = torch.zeros((2, 2), device=device)
        logits = torch.zeros((2, 2), device=device)
        rebuild_weight_matrix = torch.zeros((2, 2), device=device)

        return (loss, loss_cl, loss_rb, loss_sd, loss_mae, loss_dtw,
                loss_zcr, loss_dilate,
                metric_sd, metric_pcc, metric_si_snr,
                positives_mask, logits, rebuild_weight_matrix, pred_batch_x)

    def forward(self, x_enc, x_mark_enc, batch_x=None, mask=None):
        if self.task_name == "pretrain":
            return self.pretrain(x_enc, x_mark_enc, batch_x, mask)
        if self.task_name == "finetune":
            out = self.forecast(x_enc, x_mark_enc)
            return out[:, -self.pred_len:, :]
        return None
