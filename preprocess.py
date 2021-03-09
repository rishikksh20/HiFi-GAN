import os
import glob
import tqdm
import torch
import argparse
import numpy as np
from utils.stft import TacotronSTFT
from utils.hparams import HParam
from utils.utils import read_wav_np
from datasets.audio.pitch import Dio

def main(hp, args):
    stft = TacotronSTFT(filter_length=hp.audio.filter_length,
                        hop_length=hp.audio.hop_length,
                        win_length=hp.audio.win_length,
                        n_mel_channels=hp.audio.n_mel_channels,
                        sampling_rate=hp.audio.sampling_rate,
                        mel_fmin=hp.audio.mel_fmin,
                        mel_fmax=hp.audio.mel_fmax)

    pitch = Dio(J=hp.train.cwt_bins)
    wav_files = glob.glob(os.path.join(args.data_path, '**', '*.wav'), recursive=True)

    mel_path = os.path.join(hp.data.data_path, "mels")
    pitch_path = os.path.join(hp.data.data_path, "pitch")
    pitch_avg_path = os.path.join(hp.data.data_path, "p_avg")
    pitch_std_path = os.path.join(hp.data.data_path, "p_std")
    pitch_cwt_coefs = os.path.join(hp.data.data_path, "p_cwt_coef")

    # Create all folders
    os.makedirs(mel_path, exist_ok=True)
    os.makedirs(pitch_path, exist_ok=True)
    os.makedirs(pitch_avg_path, exist_ok=True)
    os.makedirs(pitch_std_path, exist_ok=True)
    os.makedirs(pitch_cwt_coefs, exist_ok=True)
    for wavpath in tqdm.tqdm(wav_files, desc='preprocess wav to mel'):
        sr, wav = read_wav_np(wavpath)
        assert sr == hp.audio.sampling_rate, \
            "sample rate mismatch. expected %d, got %d at %s" % \
            (hp.audio.sampling_rate, sr, wavpath)
        
        if len(wav) < hp.audio.segment_length + hp.audio.pad_short:
            wav = np.pad(wav, (0, hp.audio.segment_length + hp.audio.pad_short - len(wav)), \
                    mode='constant', constant_values=0.0)

        p, avg, std, p_coef = pitch.forward(torch.from_numpy(wav))  # shape in order - (T,) (no of utternace, ), (no of utternace, ), (10, T)

        wav = torch.from_numpy(wav).unsqueeze(0)
        mel = stft.mel_spectrogram(wav)  # mel [1, num_mel, T]

        mel = mel.squeeze(0)  # [num_mel, T]
        id = os.path.basename(wavpath).split(".")[0]
        np.save('{}/{}.npy'.format(mel_path, id), mel.numpy(), allow_pickle=False)

        np.save("{}/{}.npy".format(pitch_path, id), p, allow_pickle=False)
        np.save("{}/{}.npy".format(pitch_avg_path, id), avg, allow_pickle=False)
        np.save("{}/{}.npy".format(pitch_std_path, id), std, allow_pickle=False)
        np.save("{}/{}.npy".format(pitch_cwt_coefs, id), p_coef.reshape(-1, hp.train.cwt_bins), allow_pickle=False)
        #torch.save(mel, melpath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for config.")
    parser.add_argument('-d', '--data_path', type=str, required=True,
                        help="root directory of wav files")
    args = parser.parse_args()
    hp = HParam(args.config)

    main(hp, args)