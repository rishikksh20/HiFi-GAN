import torch
from torch import nn
from utils.utils import weights_init

class ResStack(nn.Module):
    def __init__(self, kernel, channel, padding, dilations = [1, 3, 5]):
        super(ResStack, self).__init__()
        resstack = []
        for dilation in dilations:
                resstack += [
                    nn.LeakyReLU(0.2),
                    nn.ReflectionPad1d(dilation),
                    nn.utils.weight_norm(nn.Conv1d(channel, channel, kernel_size=kernel, dilation=dilation)),
                    nn.LeakyReLU(0.2),
                    nn.ReflectionPad1d(padding),
                    nn.utils.weight_norm(nn.Conv1d(channel, channel, kernel_size=1)),
                ]
        self.resstack = nn.Sequential(*resstack)

        self.shortcut = nn.utils.weight_norm(nn.Conv1d(channel, channel, kernel_size=1))

    def forward(self, x):
      x1 = self.shortcut(x)
      x2 = self.resstack(x)
      return  x1 + x2

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.shortcut)
