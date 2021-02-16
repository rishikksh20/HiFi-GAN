from torch import nn
import torch.nn.functional as F


class PeriodDiscriminator(nn.Module):

    def __init__(self, period):
        super(PeriodDiscriminator, self).__init__()

        self.period = period
        self.discriminator = nn.ModuleList([
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv2d(1, 64, kernel_size=(5, 1), stride=(3, 1))),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv2d(64, 128, kernel_size=(5, 1), stride=(3, 1))),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv2d(128, 256, kernel_size=(5, 1), stride=(3, 1))),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv2d(256, 512, kernel_size=(5, 1), stride=(3, 1))),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv2d(512, 1024, kernel_size=(5, 1))),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.utils.weight_norm(nn.Conv2d(1024, 1, kernel_size=(3, 1))),
        ])


    def forward(self, x):
        batch_size = x.shape[0]
        pad = self.period - (x.shape[-1] % self.period)
        x = F.pad(x, (0, pad), "reflect")
        y = x.view(batch_size, -1, self.period).contiguous()
        y = y.unsqueeze(1)
        features = list()
        for module in self.discriminator:
            out = module(y)
            features.append(out)
        return features[-1], features[:-1]
