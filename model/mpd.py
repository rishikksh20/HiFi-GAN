from torch import nn
from model.period_discriminator import PeriodDiscriminator


class MPD(nn.Module):
    def __init__(self, periods=[2, 3, 5, 7, 11]):
        super(MPD, self).__init__()
        self.discriminators = nn.ModuleList([ PeriodDiscriminator(periods[0]),
                                              PeriodDiscriminator(periods[1]),
                                              PeriodDiscriminator(periods[2]),
                                              PeriodDiscriminator(periods[3]),
                                              PeriodDiscriminator(periods[4]),
                                            ])

    def forward(self, x):
        scores = list()
        feats = list()
        for key, disc in enumerate(self.discriminators):
            score, feat = disc(x)
            scores.append(score)
            feats.append(feat)
        return scores, feats


