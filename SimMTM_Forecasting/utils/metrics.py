import torch
import torch.nn as nn
import numpy as np



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
