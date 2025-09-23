import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class SignalDice(nn.Module):
    def __init__(self, eps=1e-6):
        super(SignalDice,self).__init__()
        self.eps = eps
    
    def calc_inter(self, a, b, same_sign_mat):
        a = a * same_sign_mat
        b = b * same_sign_mat
        return torch.where(a >= b, b, a)

    def calc_union(self, a, b):
        return a + b
    
    def forward(self, inputs, targets):
        device = inputs.get_device() if inputs.get_device() != -1 else "cpu"
        # Make abs value
        in_abs = torch.abs(inputs)
        tar_abs = torch.abs(targets)

        # Make Heaviside Matrix
        with torch.no_grad():
            same_sign_mat = torch.heaviside(inputs * targets, torch.tensor([0.], device=device))

        self.intersection = self.calc_inter(in_abs, tar_abs, same_sign_mat) 
        self.union        = self.calc_union(in_abs, tar_abs)
       
        return torch.mean((2 * torch.sum(self.intersection) + self.eps) / (torch.sum(self.union) + self.eps)) 
        
def pearson_correlation(pred, target, eps=1e-8):
    """
    Computes the Pearson Correlation Coefficient for a batch of signals.
    Args:
        pred (torch.Tensor): Predicted signals, shape [batch_size, sequence_length]
        target (torch.Tensor): Ground truth signals, shape [batch_size, sequence_length]
        eps (float): A small value to prevent division by zero.
    Returns:
        torch.Tensor: A scalar tensor with the mean Pearson correlation.
    """
    # 1. Center both prediction and target tensors (subtract the mean)
    pred_centered = pred - torch.mean(pred, dim=-1, keepdim=True)
    target_centered = target - torch.mean(target, dim=-1, keepdim=True)

    # 2. Compute the cosine similarity between the centered tensors.
    # This is mathematically equivalent to the Pearson correlation.
    cosine_sim = F.cosine_similarity(pred_centered, target_centered, dim=-1, eps=eps)

    # 3. Return the average correlation across the batch
    return torch.mean(cosine_sim)


def si_snr(pred, target, eps=1e-8):
    """
    Computes the Scale-Invariant Signal-to-Noise Ratio (SI-SNR) for a batch of signals.
    Args:
        pred (torch.Tensor): Predicted signals, shape [batch_size, sequence_length]
        target (torch.Tensor): Ground truth signals, shape [batch_size, sequence_length]
        eps (float): A small value to prevent division by zero.
    Returns:
        torch.Tensor: A scalar tensor with the mean SI-SNR in decibels (dB).
    """
    # 1. Ensure signals are zero-mean
    target = target - torch.mean(target, dim=-1, keepdim=True)
    pred = pred - torch.mean(pred, dim=-1, keepdim=True)

    # 2. Calculate the optimal scaling factor for the prediction
    # This is the core of the "scale-invariant" part
    s_target = torch.sum(pred * target, dim=-1, keepdim=True) * target / (torch.sum(target**2, dim=-1, keepdim=True) + eps)

    # 3. Decompose the prediction into the scaled target and the noise/error
    e_noise = pred - s_target

    # 4. Calculate the power of the scaled target and the noise
    s_target_power = torch.sum(s_target**2, dim=-1)
    e_noise_power = torch.sum(e_noise**2, dim=-1)

    # 5. Compute the SI-SNR in decibels (dB)
    snr = 10 * torch.log10((s_target_power / (e_noise_power + eps)) + eps)

    # 6. Return the average SI-SNR across the batch
    return torch.mean(snr)


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe
