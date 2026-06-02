"""Canonical SDSC (Signal Dice Similarity Coefficient) implementation.

This file is the SINGLE SOURCE OF TRUTH for SDSC after the AC-5 canonical lock
(git tag: sdsc-canonical-v1, see paper_supplement/AAAI27_metric_validation_plan.md).

Two legacy implementations diverge on three axes and remain in the repo as
DEPRECATED references only. Do not use them for new work:

  libs/metric.py:SignalDice          — H(0)=1, per-sequence reduction (dim=-1)
                                       hard variant only; soft is separate class
  SimMTM_Forecasting/utils/metrics.py:SignalDice
                                       — H(0)=0, GLOBAL reduction (no dim)
                                       alpha-switchable, but soft path runs
                                       inside torch.no_grad() → gradient is
                                       silently broken (latent bug)

Canonical choices (from AC-5):
  (i)   H(0) = 0    — conservative, matches sign(0)=0; consistent with
                      utils/metrics.py:33.
  (ii)  Reduction   — per-sequence sum(dim=-1) then mean; consistent with
                      libs/metric.py:30. This matches V1 pair-ranking semantics
                      where each pair is a separate sequence.
  (iii) Hard/soft   — single class, alpha switch (alpha=None → hard, alpha>0
                      → soft sigmoid). Soft path does NOT use torch.no_grad()
                      so SDSC can be used as a differentiable loss.
  (iv)  Numerical   — eps placed on BOTH numerator and denominator (eps in
                      both keeps the metric well-defined when |E|=|R|=0).

Returns mean over batch dimension (after per-sequence dice).

Range:
  SDSC ∈ [0, 1]   where 1 = perfect sign+magnitude overlap, 0 = no overlap.
  (Proof sketch in Appendix B of the AAAI27 submission.)
"""

import torch
import torch.nn as nn


class SignalDiceCanonical(nn.Module):
    """Canonical SDSC. Replaces both libs/metric.py and utils/metrics.py SDSC.

    Args:
        alpha: None → hard heaviside (H(0)=0). Positive float → soft sigmoid
               with sharpness alpha (gradient propagates).
        eps:   Numerical stabilizer in both numerator and denominator.
    """

    def __init__(self, alpha=None, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.eps = eps

    @staticmethod
    def _calc_intersection(a_abs, b_abs, gate):
        """Element-wise min(|a|, |b|) gated by sign-agreement matrix.

        Implements the M(E,R) = min(|E|, |R|) operator from the SDSC paper
        (Lemma B.1). The torch.where form is equivalent to torch.min(a, b)
        but is what the original codebase uses; kept for bit-identity.
        """
        a_g = a_abs * gate
        b_g = b_abs * gate
        return torch.where(a_g >= b_g, b_g, a_g)

    @staticmethod
    def _calc_union(a_abs, b_abs):
        """|E| + |R|. Always non-negative — denominator stays well-defined."""
        return a_abs + b_abs

    def forward(self, inputs, targets):
        # |E|, |R|
        in_abs = torch.abs(inputs)
        tar_abs = torch.abs(targets)

        # Sign-agreement gate.
        # Hard:  H(E·R) with H(0)=0   (canonical choice (i))
        # Soft:  sigmoid(alpha · E·R) (canonical choice (iii) — gradient flows)
        if self.alpha is None:
            with torch.no_grad():
                gate = torch.heaviside(
                    inputs * targets,
                    torch.tensor([0.0], device=inputs.device),
                )
        else:
            # NOTE: NO torch.no_grad() here. The soft path is differentiable
            # by design so the loss can backprop through the sign gate.
            gate = torch.sigmoid(inputs * targets * self.alpha)

        intersection = self._calc_intersection(in_abs, tar_abs, gate)
        union = self._calc_union(in_abs, tar_abs)

        # Per-sequence dice over the last axis, then batch mean.
        # (Canonical choice (ii) — matches V1 pair-ranking semantics.)
        per_seq_dice = (
            2.0 * torch.sum(intersection, dim=-1) + self.eps
        ) / (torch.sum(union, dim=-1) + self.eps)

        return torch.mean(per_seq_dice)


class SignalDiceLossCanonical(nn.Module):
    """Differentiable loss form: 1 - SDSC. Pairs naturally with the soft gate.

    For training, use alpha=10.0 (best validated in ICLR rebuttal alpha
    sweep). For evaluation, use SignalDiceCanonical(alpha=None) directly.
    """

    def __init__(self, alpha=10.0, eps=1e-6):
        super().__init__()
        self.sdsc = SignalDiceCanonical(alpha=alpha, eps=eps)

    def forward(self, inputs, targets):
        return 1.0 - self.sdsc(inputs, targets)
