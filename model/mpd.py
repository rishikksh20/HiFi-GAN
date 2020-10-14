from torch import nn

class MPD(nn.Module):

  def __init__(self):
    super(MPD, self).__init__()
    layer = []

    for l in range(4):
      out = int(2**(5+l+1))
      layer =+ [
                nn.Conv2d(1, out, kernel_size=(5, 1), stride=(3, 1))
                nn.LeakyReLU(0.2)
      ]
    self.layer = nn.Sequential(*layer)
    self.output = nn.Sequential(
        nn.Conv2d(out, 1024, kernel_size=(5, 1)),
        nn.LeakyReLU(0.2),
        nn.Conv2d(1024, 1, kernel_size=(3, 1))
        )
  
  def forward(self, x):
    out1 = self.layer(x)
    return self.output(out1)