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
            same_sign_mat = torch.heaviside(inputs * targets, torch.tensor([1.], device=device))

        self.intersection = self.calc_inter(in_abs, tar_abs, same_sign_mat) 
        self.union        = self.calc_union(in_abs, tar_abs)
       
        return torch.mean((2 * torch.sum(self.intersection, dim=-1) + self.eps) / (torch.sum(self.union, dim=-1) + self.eps)) 
        
