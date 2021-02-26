from __future__ import absolute_import, division, print_function, unicode_literals
import os
import torch
import argparse
from scipy.io.wavfile import write
import numpy as np
from model.generator import Generator
import json
from utils.hparams import HParam, load_hparam_str
from denoiser import Denoiser
from model.pqmf import PQMF
from torch import nn


MAX_WAV_VALUE = 32768.0


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


class Generator_torch_script(nn.Module):
    def __init__(self, hp, args):
        super(Generator_torch_script, self).__init__()
        self.generator = Generator(hp.audio.n_mel_channels, hp.model.out_channels, hu=384, ku=[8, 8, 4, 4], kr=[3, 7, 11], Dr=[1, 3, 5])
        self.pqmf = PQMF()
        self.checkpoint = torch.load(args.checkpoint_path)

    def forward(self, x):
        self.generator.load_state_dict(self.checkpoint['model_g'])
        self.generator.eval()
        audioG = self.generator(x)
        out = self.pqmf.synthesis(audioG)
        return out


def main(args):
    #checkpoint = torch.load(args.checkpoint_path)
    if args.config is not None:
        hp = HParam(args.config)
    else:
        hp = load_hparam_str(checkpoint['hp_str'])

    model = Generator_torch_script(hp, args).cuda()

    #model = Generator(hp.audio.n_mel_channels).cuda()
    #model.remove_weight_norm()

    with torch.no_grad():
        mel = torch.from_numpy(np.load(args.input))
        if len(mel.shape) == 2:
            mel = mel.unsqueeze(0)
        mel = mel.cuda()
        #zero = torch.full((1, 80, 10), -11.5129).to(mel.device)
        #mel = torch.cat((mel, zero), dim=2)
        hifigan_trace = torch.jit.trace(model, mel)
        #print(state_dict_g.keys())
        hifigan_trace.save("{}/hifigan_{}.pt".format(args.out, args.name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=None, required=True,
                        help="yaml file for config. will use hp_str from checkpoint if not given.")
    parser.add_argument('-p', '--checkpoint_path', type=str, required=True,
                        help="path of checkpoint pt file for evaluation")
    parser.add_argument('-i', '--input', type=str, required=True,
                        help="directory of mel-spectrograms to invert into raw audio. ")
    parser.add_argument('-o', '--out', type=str, required=True,
                        help="path of output pt file")
    parser.add_argument('-n', '--name', type=str, required=True,
                        help="name of the output file")
    args = parser.parse_args()

    main(args)
