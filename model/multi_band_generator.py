import torch

from model.generator import Generator
from model.pqmf import PQMF


class MultibandGenerator(Generator):
    def __init__(self, input_channel=80, hu=512, ku=[16, 16, 4, 4], kr=[3, 7, 11], Dr=[1, 3, 5]):
        super(MultibandGenerator, self).__init__(input_channel=input_channel, hu=hu, ku=ku, kr=kr, Dr =Dr)
        self.pqmf_layer = PQMF(N=4, taps=62, cutoff=0.15, beta=9.0)

    def pqmf_analysis(self, x):
        return self.pqmf_layer.analysis(x)

    def pqmf_synthesis(self, x):
        return self.pqmf_layer.synthesis(x)

    @torch.no_grad()
    def inference(self, cond_features):
        cond_features = cond_features.to(self.layers[1].weight.device)
        cond_features = torch.nn.functional.pad(
            cond_features,
            (self.inference_padding, self.inference_padding),
            'replicate')
        return self.pqmf_synthesis(self.layers(cond_features))
