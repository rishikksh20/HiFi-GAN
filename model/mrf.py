from model.resstack import ResStack
from torch import nn
from utils.utils import weights_init

class MRF(nn.Module):
    def __init__(self, kernels, channel, dilations = [[1,1], [3,1], [5,1]]):
        super(MRF, self).__init__()
        self.resblock1 = ResStack(kernels[0], channel, 0)
        self.resblock2 = ResStack(kernels[1], channel, 6)
        self.resblock3 = ResStack(kernels[2], channel, 12)

    def forward(self, x):
      x1 = self.resblock1(x)
      x2 = self.resblock2(x)
      x3 = self.resblock3(x)
      return x1 + x2 + x3

    def remove_weight_norm(self):
        self.resblock1.remove_weight_norm()
        self.resblock2.remove_weight_norm()
        self.resblock3.remove_weight_norm()
