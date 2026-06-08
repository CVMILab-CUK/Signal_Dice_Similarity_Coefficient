"""V1 pre-registered co-baselines for SDSC metric validation (AAAI27).

These baselines are NOT used as training losses by the main 200-cell sweep.
They exist solely as metrics in the V1 pair-ranking experiment (plan v5 AC-2).

Pre-registered ordering on family (g) — sign-preserving structural damage:
    SDSC ρ > 2-bit μ-law MSE ρ > 1-bit MSE ρ > ZCR ρ
"""

from .zcr_diff import DiffZCRLoss
from .quantized_mse import one_bit_mse, two_bit_mu_law_mse, MuLawQuantize

__all__ = ["DiffZCRLoss", "one_bit_mse", "two_bit_mu_law_mse", "MuLawQuantize"]
