import torch
import torch.nn as nn
from libs.metric import SignalDice, SoftSignalDice

def l1(x, y):
    return torch.abs(x-y)


def l2(x, y):
    return torch.pow((x-y), 2)


class SignalDiceLoss(nn.Module):

    def __init__(self,  eps=1e-6, soft=True, alpha=100):
        super(SignalDiceLoss, self).__init__()
        self.eps  = eps
        if soft:
            self.sdsc = SoftSignalDice(eps, alpha=alpha)
        else:
            self.sdsc = SignalDice(eps)
    
    def forward(self, inputs, targets):
        sdsc_value = self.sdsc(inputs, targets)

        return 1 - sdsc_value