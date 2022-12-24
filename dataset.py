import torch
from torch import nn

import torchaudio
import librosa
from scipy.io.wavfile import read
from scipy.io.wavfile import write

import os
import numpy as np
from torch.utils.data import Dataset

import random

class MelSpectrogram(torch.nn.Module):
    def __init__(self, config):
        super(MelSpectrogram, self).__init__()

        self.config = config

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sr,
            win_length=config.win_length,
            hop_length=config.hop_length,
            n_fft=config.n_fft,
            f_min=config.f_min,
            f_max=config.f_max,
            n_mels=config.n_mels,
            center=False
        )

        # The is no way to set power in constructor in 0.5.0 version.
        self.mel_spectrogram.spectrogram.power = config.power

        # Default `torchaudio` mel basis uses HTK formula. In order to be compatible with WaveGlow
        # we decided to use Slaney one instead (as well as `librosa` does by default).
        mel_basis = librosa.filters.mel(
            sr=config.sr,
            n_fft=config.n_fft,
            n_mels=config.n_mels,
            fmin=config.f_min,
            fmax=config.f_max
        ).T
        
        self.mel_spectrogram.mel_scale.fb.copy_(torch.tensor(mel_basis))
        
    def forward(self, audio: torch.Tensor): #  -> torch.Tensor
        """
        :param audio: Expected shape is [B, T]
        :return: Shape is [B, n_mels, T']
        """
        
        audio = audio.squeeze(1)
        audio = torch.nn.functional.pad(audio.unsqueeze(1), (int((self.config.n_fft-self.config.hop_length)/2), int((self.config.n_fft-self.config.hop_length)/2)), mode='reflect').squeeze(1)
        mel = self.mel_spectrogram(audio).clamp_(min=1e-5).log_()

        return mel
        
    
class MelDataset(Dataset):
    def __init__(self, wav_dir, segment_size):
        self.paths_to_wav = sorted([wav_dir + '/' + name for name in os.listdir(wav_dir)])
        self.segment_size = segment_size
        
    def __len__(self):
        return len(self.paths_to_wav)
    
    def __getitem__(self, idx):
        sampling_rate, audio = read(self.paths_to_wav[idx])
        audio = torch.FloatTensor(audio.astype(np.float32))
        if self.segment_size > 0:
            if audio.shape[-1] < self.segment_size:
                audio = F.pad(audio, (0, self.segment_size - audio.shape[-1]))
            elif audio.shape[-1] > self.segment_size:
                start = np.random.randint(0, audio.shape[-1] - self.segment_size)
                audio = audio[..., start:start+self.segment_size]
        return audio