import torch as t
import torch
import torch.nn as nn
import numpy as np

import pysdtw
from pysdtw import SoftDTW

from .metrics import SignalDice


class SignalDiceLoss(nn.Module):

    def __init__(self,  eps=1e-6):
        super(SignalDiceLoss, self).__init__()
        self.sdsc = SignalDice(eps)
        self.eps  = eps
    
    def forward(self, inputs, targets):
        sdsc_value = self.sdsc(inputs, targets)
        return 1 - sdsc_value

class mae_loss(nn.Module):
    def __init__(self):
        super(mae_loss, self).__init__()

    def forward(self, inputs, targets):
        return torch.mean(torch.abs(inputs, targets))

class dtw_loss(nn.Module):
    def __init__(self, approx=True, gamma=1.0):
        super(dtw_loss, self).__init__()

        if approx:
            fun = pysdtw.distance.pairwise_l2_squared
            self.dtw = SoftDTW(gamma = gamma, dist_func=fun, use_cuda=False)
        else:
            self.dtw = self.dtw_distance_torch

    def dtw_distance_torch(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute DTW distance for a batch of 1D signal pairs.
        
        Args:
            x: tensor of shape (B, T)
            y: tensor of shape (B, T)
            
        Returns:
            dtw_distances: tensor of shape (B,)
        """
        B, T = inputs.shape
        dtw_distances = []

        for b in range(B):
            x_b = inputs[b]
            y_b = targets[b]
            T1, T2 = len(x_b), len(y_b)

            dtw = torch.full((T1 + 1, T2 + 1), float('inf'), device=inputs.device)
            dtw[0, 0] = 0.0

            for i in range(1, T1 + 1):
                for j in range(1, T2 + 1):
                    cost = torch.abs(x_b[i - 1] - y_b[j - 1])
                    dtw[i, j] = cost + torch.min(
                        dtw[i - 1, j],
                        dtw[i, j - 1],
                        dtw[i - 1, j - 1]
                    )

            dtw_distances.append(dtw[T1, T2] / T1)  # normalize

        return torch.stack(dtw_distances)  # (B,)

    
    def forward(self, inputs, targets):
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