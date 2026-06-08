"""Differentiable Zero Crossing Rate (ZCR) loss — speech-literature soft-sign
relaxation.

Reference: standard `tanh(α · x)` ≈ `sign(x)` as α → ∞ is widely used as a
differentiable proxy for sign-based losses in speech (e.g., autoregressive
vocoders that match phase via differentiable sign agreement). The same
relaxation is reused here as the literature-derived ZCR baseline for V1.

Why "literature-derived" matters: ICML 4Zmu's reject critique was that SDSC
is just "sign quantization + magnitude weighting." A reviewer can dismiss
an author-invented ZCR baseline as a strawman. By using a published soft-sign
relaxation form, we close that escape hatch.

V1 prediction:
    Family (g) sign-preserving structural damage → SDSC ρ > ZCR ρ
        because SDSC's magnitude weighting captures structural information that
        sign-only ZCR cannot see.

The hard ZCR objective ``mean((sign(E) - sign(R))**2)`` is non-differentiable;
``mean((tanh(αE) - tanh(αR))**2)`` is a differentiable approximation that
converges to the hard form as α → ∞.
"""

import torch
import torch.nn as nn


class DiffZCRLoss(nn.Module):
    """Differentiable sign-only mismatch via tanh soft-sign relaxation.

    Args:
        alpha: tanh sharpness. α=10 matches the SDSC soft-gate convention used
            in `SignalDiceCanonical(alpha=10.0)`.

    The output equals 0 if ``sign(pred) == sign(target)`` everywhere (modulo
    soft-sign saturation), and approaches 4 when signs disagree everywhere.
    """

    def __init__(self, alpha: float = 10.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # tanh(α · x) ∈ (−1, 1); saturates to sign(x) for |x| ≫ 1/α
        p_sign = torch.tanh(self.alpha * pred)
        t_sign = torch.tanh(self.alpha * target)
        return torch.mean((p_sign - t_sign) ** 2)


def hard_zcr_metric(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Non-differentiable hard ZCR for evaluation (V1 metric reporting only).

    Returns fraction of positions where signs DISAGREE (∈ [0, 1]).
    """
    with torch.no_grad():
        return torch.mean((torch.sign(pred) != torch.sign(target)).float())
