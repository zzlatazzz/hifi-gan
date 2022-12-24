from dataclasses import dataclass

import torch
from torch import nn

import torchaudio
import librosa  


@dataclass
class ModelConfig:
    leaky_relu_slope = 0.1

    upsample_kernel_sizes = [16, 16, 4, 4]
    upsample_initial_channel = 128
    resblock_kernel_sizes = [3, 7, 11]
    resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]

    periods = [2, 3, 5, 7, 11]

    
@dataclass
class TrainConfig:
    train_wav_path = './data/LJSpeech-1.1/wavs'
    test_wav_path = './test_wavs'
    
    checkpoint_dir = 'checkpoints'
    checkpoint_path = 'checkpoints/checkpoint.s'
    wandb_project = 'dla-hw4-hifi'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    segment_size = 8192
    
    batch_size = 98
    learning_rate = 1e-3
    adam_b1 = 0.8
    adam_b2 = 0.99
    gamma = 0.999
    
    num_workers = 32
    
    seed = 42

    last_epoch = -1
    num_epochs = 50

    log_step = 20
    freq_save_model = 100
    
    feature_mul = 2
    mel_mul = 45


@dataclass
class MelSpectrogramConfig:
    sr: int = 22050
    win_length: int = 1024
    hop_length: int = 256
    n_fft: int = 1024
    f_min: int = 0
    f_max: int = 8000
    n_mels: int = 80
    power: float = 1.0

    # value of melspectrograms if we fed a silence into `MelSpectrogram`
    pad_value: float = -11.5129251
        

