import os
import glob
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader

from utils.utils import read_wav_np


def create_dataloader(hp, args, train):
    dataset = MelFromDisk(hp, args, train)

    if train:
        return DataLoader(dataset=dataset, batch_size=hp.train.batch_size, shuffle=True,
            num_workers=0, pin_memory=True, drop_last=True)
    else:
        return DataLoader(dataset=dataset, batch_size=1, shuffle=False,
            num_workers=0, pin_memory=False, drop_last=False)

def get_dataset_filelist(file, wav_dir):
    with open(file, 'r', encoding='utf-8') as fi:
        training_files = [os.path.join(wav_dir, x.split('|')[0])
                          for x in fi.read().split('\n') if len(x) > 0]
    return training_files

class MelFromDisk(Dataset):
    def __init__(self, hp, args, train):
        self.hp = hp
        self.args = args
        self.train = train
        self.path = hp.data.train if train else hp.data.valid
        self.wav_list = get_dataset_filelist(self.path, hp.data.wav_dir)
        #print("Wavs path :", self.path)
        #print(self.hp.data.mel_path)
        #print("Length of wavelist :", len(self.wav_list))
        self.mel_segment_length = hp.audio.segment_length // hp.audio.hop_length + 2
        self.mapping = [i for i in range(len(self.wav_list))]

    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, idx):

        if self.train:
            idx1 = idx
            idx2 = self.mapping[idx1]
            return self.my_getitem(idx1), self.my_getitem(idx2)
        else:
            return self.my_getitem(idx)

    def shuffle_mapping(self):
        random.shuffle(self.mapping)

    def my_getitem(self, idx):
        wavpath = self.wav_list[idx]
        id = os.path.basename(wavpath).split(".")[0]
        mel_path = os.path.join(self.hp.data.data_path, "mels")
        mel_path = "{}/{}.npy".format(mel_path, id)

        if self.hp.train.cwt:
            conditional_path = os.path.join(self.hp.data.data_path, "p_cwt_coef")
        else:
            conditional_path = os.path.join(self.hp.data.data_path, "pitch")
        conditional_path = "{}/{}.npy".format(conditional_path, id)


        sr, audio = read_wav_np(wavpath)
        if len(audio) < self.hp.audio.segment_length + self.hp.audio.pad_short:
            audio = np.pad(audio, (0, self.hp.audio.segment_length + self.hp.audio.pad_short - len(audio)), \
                    mode='constant', constant_values=0.0)

        audio = torch.from_numpy(audio).unsqueeze(0)
        # mel = torch.load(melpath).squeeze(0) # # [num_mel, T]

        mel = torch.from_numpy(np.load(mel_path))
        c = torch.from_numpy(np.load(conditional_path))
        if self.train:
            max_mel_start = mel.size(1) - self.mel_segment_length
            mel_start = random.randint(0, max_mel_start)
            mel_end = mel_start + self.mel_segment_length
            mel = mel[:, mel_start:mel_end]
            if self.hp.train.cwt:
                #print("Shape of c :", c.shape)
                c = c[mel_start:mel_end, :].T
            else:
                c = c.unsqueeze(0)
                c = c[:, mel_start:mel_end]

            #print("C :", c.shape)
            audio_start = mel_start * self.hp.audio.hop_length
            audio = audio[:, audio_start:audio_start+self.hp.audio.segment_length]

        audio = audio + (1/32768) * torch.randn_like(audio)
        return mel, audio, c
