from model.resstack import ResStack
from torch import nn
from utils.utils import weights_init

class MRF(nn.Module):
    def __init__(self, kernels, channel, dilations):
        super(MRF, self).__init__()
        self.resblock1 = ResStack(kernels[0], channel, dilations)
        self.resblock2 = ResStack(kernels[1], channel, dilations)
        self.resblock3 = ResStack(kernels[2], channel, dilations)

    def forward(self, x):
        return self.resblock1(x) + self.resblock2(x) + self.resblock3(x)