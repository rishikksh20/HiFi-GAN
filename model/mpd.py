from torch import nn
import torch.nn.functional as F

class MPD(nn.Module):

  def __init__(self, periods, segment_length=16000):
    super(MPD, self).__init__()
    layer = []
    self.periods = periods
    self.pad = segment_length%periods
    inp = 1
    for l in range(4):
       
      out = int(2**(5+l+1))
      layer += [
                nn.Conv2d(inp, out, kernel_size=(5, 1), stride=(3, 1)),
                nn.LeakyReLU(0.2)
      ]
      inp = out
    self.layer = nn.Sequential(*layer)
    self.output = nn.Sequential(
        nn.Conv2d(out, 1024, kernel_size=(5, 1)),
        nn.LeakyReLU(0.2),
        nn.Conv2d(1024, 1, kernel_size=(3, 1))
        )
  
  def forward(self, x):
    batch_size = x.shape[0]
    x = F.pad(x, (0, self.pad))
    y = x.view(batch_size,-1, self.periods).contiguous()
    y = y.unsqueeze(1)
    out1 = self.layer(y)
    return self.output(out1)