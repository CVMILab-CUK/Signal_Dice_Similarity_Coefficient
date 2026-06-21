import torch as t
import torch
import torch.nn as nn
import numpy as np

import pysdtw
from pysdtw import SoftDTW

from .metrics import SignalDice

from .soft_dtw_cuda import SoftDTW


class SignalDiceLoss(nn.Module):

    def __init__(self,  alpha = None, eps=1e-6):
        super(SignalDiceLoss, self).__init__()
        self.sdsc = SignalDice(eps=eps, alpha=alpha)
        self.eps  = eps
    
    def forward(self, inputs, targets):
        sdsc_value = self.sdsc(inputs, targets)
        return 1 - sdsc_value

class mae_loss(nn.Module):
    def __init__(self):
        super(mae_loss, self).__init__()

    def forward(self, inputs, targets):
        return torch.mean(torch.abs(inputs - targets))


class _SignSTE(torch.autograd.Function):
    """Hard sign in the forward pass, identity (straight-through) gradient in the backward pass.

    This is the canonical 1-bit quantization gradient estimator referenced in the
    Van Vleck arcsine literature; see Bengio, Leonard & Courville (2013) "Estimating
    or Propagating Gradients Through Stochastic Neurons for Conditional Computation".
    """

    @staticmethod
    def forward(ctx, x):
        return torch.sign(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through: pass gradient unchanged.
        return grad_output


def _sign_ste(x):
    return _SignSTE.apply(x)


class ZCRLoss(nn.Module):
    """Sign-only ablation of SDSC (US-002 / F3).

    SDSC has two ingredients: a sign-agreement gate and a min(|E|,|R|) magnitude weight.
    ZCRLoss isolates the sign gate by dropping the magnitude weight, so the comparison
    only depends on whether ``inputs`` and ``targets`` share signs sample-by-sample.

    Form:
        agreement = sigmoid(alpha * sign_STE(E) * sign_STE(R))
        ZCR_loss  = 1 - mean(agreement)

    Why straight-through estimator? ``torch.sign`` is exact (preserves magnitude
    invariance perfectly), but its gradient is zero almost everywhere, so a plain
    hard-sign loss is not trainable. The STE keeps the forward exact (sign in
    {-1, +1, 0}) and routes gradients through as identity, so the model can still
    learn. This matches the 1-bit-quantization framing requested by ICML reviewer
    4Zmu and the Van Vleck arcsine literature on coarse quantization.

    Magnitude invariance is exact for all non-zero scalings: ``sign(c*x) = sign(x)``
    for any ``c > 0``.
    """

    def __init__(self, alpha=10.0):
        super(ZCRLoss, self).__init__()
        self.alpha = alpha

    def forward(self, inputs, targets):
        sign_e = _sign_ste(inputs)
        sign_r = _sign_ste(targets)
        agreement = torch.sigmoid(self.alpha * sign_e * sign_r)
        return 1.0 - torch.mean(agreement)

    @staticmethod
    def hard_metric(inputs, targets):
        """Non-differentiable sign-agreement rate (for logging only)."""
        return torch.mean((inputs * targets > 0).float())


class DILATELoss(nn.Module):
    """DILATE = Shape + Temporal distortion loss (Le Guen & Thome, NeurIPS 2019, US-003 / F2).

    L_DILATE = gamma_dilate * L_shape + (1 - gamma_dilate) * L_temporal
        L_shape    = Soft-DTW(pred, target)
        L_temporal = sum( soft_alignment_path * Omega ) / N^2
        Omega[i,j] = (i - j)^2 / N^2      # squared temporal-distortion penalty

    The temporal term uses the *soft alignment path* returned by ``PathDTWBatch``,
    not the gradient of Soft-DTW w.r.t. the cost matrix. These are different
    quantities; using the latter (a v1 mistake) would not produce the correct TDI.

    Inputs:
        pred:   (B, T, C) or (B, T)
        target: (B, T, C) or (B, T)

    Returns:
        scalar mean DILATE loss over the batch.
    """

    def __init__(self, gamma_dilate=0.5, gamma_sdtw=0.01, use_cuda=True):
        super(DILATELoss, self).__init__()
        self.gamma_dilate = gamma_dilate
        self.gamma_sdtw = gamma_sdtw
        # SoftDTW from soft_dtw_cuda (Maghoumi implementation, already in repo)
        self.sdtw = SoftDTW(use_cuda=use_cuda, gamma=gamma_sdtw, normalize=False)
        self._omega_cache = {}  # cached per (N, device) pair

    @staticmethod
    def _pairwise_l2(a, b):
        """(N, d) x (N, d) -> (N, N) pairwise squared L2 distance, clamped >= 0."""
        a_norm = (a * a).sum(-1, keepdim=True)        # (N, 1)
        b_norm = (b * b).sum(-1, keepdim=True).T      # (1, N)
        cross = a @ b.T                                # (N, N)
        return torch.clamp(a_norm + b_norm - 2.0 * cross, min=0.0)

    def _omega(self, N, device, dtype):
        key = (N, device, dtype)
        if key in self._omega_cache:
            return self._omega_cache[key]
        idx = torch.arange(1, N + 1, dtype=dtype, device=device)
        Omega = (idx.unsqueeze(0) - idx.unsqueeze(1)) ** 2
        self._omega_cache[key] = Omega
        return Omega

    def forward(self, pred, target):
        from .path_soft_dtw import PathDTWBatch  # local import to avoid heavy numba load at module init

        if pred.dim() == 2:
            pred = pred.unsqueeze(-1)
            target = target.unsqueeze(-1)
        # Now (B, T, C). Average DILATE over channels by reshaping into batch.
        B, T, C = pred.shape
        pred_bc = pred.permute(0, 2, 1).reshape(B * C, T, 1)
        target_bc = target.permute(0, 2, 1).reshape(B * C, T, 1)
        device = pred.device
        dtype = pred.dtype

        # ----- L_shape via Soft-DTW
        loss_shape = self.sdtw(pred_bc, target_bc).mean()

        # ----- L_temporal via soft-alignment path
        # Build (B*C, T, T) pairwise distance matrices.
        D = torch.zeros((B * C, T, T), device=device, dtype=dtype)
        for k in range(B * C):
            D[k] = self._pairwise_l2(target_bc[k], pred_bc[k])

        # PathDTWBatch returns (T, T) averaged path; we recompute per-batch for proper averaging.
        # The original DILATE uses mean across batch internally -- we keep that contract.
        path = PathDTWBatch.apply(D, self.gamma_sdtw)  # (T, T)
        Omega = self._omega(T, device, dtype)          # (T, T)
        loss_temporal = (path * Omega).sum() / (T * T)

        return self.gamma_dilate * loss_shape + (1.0 - self.gamma_dilate) * loss_temporal


class DTWLoss(nn.Module):
    """
    Soft-DTW ?????? ??? ???? DTW ?? ??.
    ?? ??: (?? ??, ??? ??, ??/?? ?)
    """
    def __init__(self, gamma=1.0, normalize=False, use_cuda=True):
        super(DTWLoss, self).__init__()
        # GPU? ???? ???? ?? ??? ?????.
        self.dtw_computer = SoftDTW(use_cuda=use_cuda, gamma=gamma, normalize=normalize)

    def forward(self, pred, true):
        # ?????? ?? ??? ?? DTW? ? ?? ?????.
        loss = self.dtw_computer(pred, true)
        return loss.mean() # ?? ??? ?? ??? ??
    
class dtw_loss(nn.Module):
    def __init__(self, approx=True, gamma=1.0, use_cuda=True):
        super(dtw_loss, self).__init__()

        if approx:
            fun = pysdtw.distance.pairwise_l2_squared
            self.dtw = SoftDTW(gamma = gamma, dist_func=fun, use_cuda=use_cuda)
        else:
            self.dtw = self.dtw_distance_torch
    
    def divergence(self, x, y):
        loss_xy = self.dtw(x, y)
        loss_xx = self.dtw(x, x)
        loss_yy = self.dtw(y, y)
        divergence = loss_xy - 0.5 * (loss_xx + loss_yy)
        return divergence.mean()

    def dtw_distance_torch(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute DTW distance for a batch of 1D signal pairs.
        
        Args:
            x: tensor of shape (B, C, T)
            y: tensor of shape (B, C, T)
            
        Returns:
            dtw_distances: tensor of shape (B,)
        """
        print(inputs.shape)
        B, C, T = inputs.shape
        inputs = inputs.reshape(B*C, T)
        targets = targets.reshape(B*C, T)

        dtw_distances = []
        for b in range(B*C):
            x_b = inputs[b]
            y_b = targets[b]
            T1, T2 = len(x_b), len(y_b)

            dtw = torch.full((T1 + 1, T2 + 1), float('inf'), device=inputs.device)
            dtw[0, 0] = 0.0

            for i in range(1, T1 + 1):
                for j in range(1, T2 + 1):
                    cost = torch.abs(x_b[i - 1] - y_b[j - 1])
                    dtw[i, j] = cost + torch.min(
                        torch.stack([dtw[i - 1, j],
                        dtw[i, j - 1],
                        dtw[i - 1, j - 1]])
                    )

            dtw_distances.append(dtw[T1, T2] / T1)  # normalize

        return torch.stack(dtw_distances).contiguous().view(B, C).mean(dim=1)  # (B, C)

    
    def forward(self, inputs, targets, div=True):
        if div:
            return self.divergence(inputs, targets)
        return self.dtw(inputs, targets).mean()


def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result


class mape_loss(nn.Module):
    def __init__(self):
        super(mape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        weights = divide_no_nan(mask, target)
        return t.mean(t.abs((forecast - target) * weights))


class smape_loss(nn.Module):
    def __init__(self):
        super(smape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        return 200 * t.mean(divide_no_nan(t.abs(forecast - target), t.abs(forecast.data) + t.abs(target.data)) * mask)


class mase_loss(nn.Module):
    def __init__(self):
        super(mase_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

        :param insample: Insample values. Shape: batch, time_i
        :param freq: Frequency value
        :param forecast: Forecast values. Shape: batch, time_o
        :param target: Target values. Shape: batch, time_o
        :param mask: 0/1 mask. Shape: batch, time_o
        :return: Loss value
        """
        masep = t.mean(t.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return t.mean(t.abs(target - forecast) * masked_masep_inv)


class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """

    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum