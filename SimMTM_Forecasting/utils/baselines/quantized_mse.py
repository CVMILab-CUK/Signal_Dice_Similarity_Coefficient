"""1-bit and 2-bit μ-law quantized MSE — V1 baselines for the
SDSC ≠ "just sign quantization + weighting" defense.

These quantization-based MSE forms span the granularity axis between
sign-only (ZCR, 1-bit) and full continuous (MSE, ∞-bit). SDSC sits in
this space as a continuous sign-gated similarity.

Pre-registered V1 ordering on family (g):
    SDSC ρ > 2-bit μ-law MSE ρ > 1-bit MSE ρ > ZCR ρ
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn


def _channelwise_zscore(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-sequence (last-axis) z-score normalization.

    Matches the canonical V1 pair protocol: pairs are constructed on
    z-normalized source signals so amplitude bias does not confound
    quantization-based metrics.
    """
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True) + eps
    return (x - mean) / std


def one_bit_mse(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """1-bit sign-quantized MSE.

    Quantize ``pred`` and ``target`` to ``sign(z(x)) ∈ {−1, +1}`` (with
    sign(0) := 0 by torch convention), then compute MSE. Tests the
    hypothesis that "SDSC = sign agreement" without magnitude weighting.

    V1 prediction: on family (g) — sign-preserving structural damage — this
    metric should be blind because signs are unchanged. SDSC should win.
    """
    p = torch.sign(_channelwise_zscore(pred, eps))
    t = torch.sign(_channelwise_zscore(target, eps))
    return torch.mean((p - t) ** 2)


class MuLawQuantize(nn.Module):
    """μ-law companding + uniform quantization.

    Standard μ-law:
        F(x) = sign(x) · log(1 + μ|x|) / log(1 + μ)
    Inverse:
        F⁻¹(y) = sign(y) · ((1+μ)^|y| − 1) / μ

    Input is first scaled to ``[−1, 1]`` per-sequence (using max|x|), then
    companded, then uniformly quantized to ``levels`` bins. Output is the
    discrete bin index ∈ [0, levels−1] cast back to float.

    Args:
        levels: quantization levels (2-bit → 4, 3-bit → 8, ...).
        mu: μ-law parameter (default 255 = G.711 standard).
    """

    def __init__(self, levels: int = 4, mu: float = 255.0):
        super().__init__()
        self.levels = levels
        self.mu = mu
        self.register_buffer(
            "edges",
            torch.linspace(-1.0, 1.0, levels + 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Scale to [-1, 1] per sequence (last axis)
        max_abs = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        x_n = x / max_abs

        # μ-law compand
        log_1pmu = math.log1p(self.mu)
        x_mu = torch.sign(x_n) * torch.log1p(self.mu * x_n.abs()) / log_1pmu

        # Uniform quantization into `levels` bins. bucketize returns 0..levels.
        idx = torch.bucketize(x_mu, self.edges) - 1
        idx = idx.clamp(0, self.levels - 1).float()
        return idx


def two_bit_mu_law_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE between 2-bit (4-level) μ-law quantized signals.

    Intermediate granularity between 1-bit sign and continuous MSE. Tests
    whether structural information appears at 2 bits before reaching SDSC's
    sign-gated continuous form.

    V1 prediction: on family (g), 2-bit μ-law MSE captures some of the
    magnitude-driven structure that 1-bit and ZCR miss, but less than SDSC.
    Pre-registered: ZCR < 1-bit < 2-bit μ-law < SDSC (by ρ on g).
    """
    q = MuLawQuantize(levels=4)
    if pred.device != q.edges.device:
        q = q.to(pred.device)
    return torch.mean((q(pred) - q(target)) ** 2)
