import torch
from torch import nn
from utils.utils import weights_init

class ResStack(nn.Module):
    def __init__(self, kernel, channel, dilations):
        super(ResStack, self).__init__()
        resstack = []
        for d in dilations:
            resstack += [
                nn.LeakyReLU(0.2),
                nn.ReflectionPad1d(d),
                nn.utils.weight_norm(nn.Conv1d(channel, channel, kernel_size=kernel, dilation=d))
            ]
        self.resstack = nn.Sequential(*resstack)

        self.shortcut = nn.utils.weight_norm(nn.Conv1d(channel, channel, kernel_size=1))

    def forward(self, x):
        return self.shortcut(x) + self.block(x)

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.shortcut)
