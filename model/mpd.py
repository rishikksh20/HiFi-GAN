from torch import nn
from model.period_discriminator import PeriodDiscriminator


class MPD(nn.Module):
    def __init__(self, periods=[2, 3, 5, 7, 11], segment_length=16000):
        super(MPD, self).__init__()
        self.mpd1 = PeriodDiscriminator(periods[0])
        self.mpd2 = PeriodDiscriminator(periods[1])
        self.mpd3 = PeriodDiscriminator(periods[2])
        self.mpd4 = PeriodDiscriminator(periods[3])
        self.mpd5 = PeriodDiscriminator(periods[4])

    def forward(self, x):
        out1 = self.mpd1(x)
        out2 = self.mpd2(x)
        out3 = self.mpd3(x)
        out4 = self.mpd4(x)
        out5 = self.mpd5(x)
        return out1, out2, out3, out4, out5


