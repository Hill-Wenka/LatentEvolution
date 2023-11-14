import torch
import torch.nn as nn
import torch.nn.functional as F


class MutationEffectLoss(nn.Module):
    def __init__(self, add_noise=False, noise_std=0.1):
        super(MutationEffectLoss, self).__init__()
        self.add_noise = add_noise
        self.noise_std = noise_std

    def forward(self, pred, target):
        if self.add_noise:
            target = target + torch.randn_like(torch) * self.noise_std
        return F.mse_loss(pred.reshape(-1), target.reshape(-1))
