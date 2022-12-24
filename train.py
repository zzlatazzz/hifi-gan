#download LjSpeech
import os

if not os.path.exists('data'):
    !wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 -o /dev/null
    !mkdir data
    !tar -xvf LJSpeech-1.1.tar.bz2 >> /dev/null
    !mv LJSpeech-1.1 data/LJSpeech-1.1import random

import random

import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

import torchaudio
import itertools

import os
import librosa
import pandas as pd
from tqdm import tqdm
import numpy as np

from dataclasses import dataclass
from collections import OrderedDict

from configs import ModelConfig, TrainConfig, MelSpectrogramConfig
from models import Generator, MPDiscriminator, MSDiscriminator
from losses import generator_loss, discriminator_loss, feature_loss, mel_loss
from dataset import MelDataset, MelSpectrogram
from wandbwriter import WanDBWriter

def save_checkpoint(path, generator, mpd, msd):
    torch.save({
        'generator': generator.state_dict(),
        'mpd': mpd.state_dict(),
        'msd': msd.state_dict()}, path)
    
def load_checkpoint(path, generator, mpd, msd):
    full_ckpt = torch.load(path)
    generator.load_state_dict(full_ckpt['generator'])
    mpd.load_state_dict(full_ckpt['mpd'])
    msd.load_state_dict(full_ckpt['msd'])
    return generator, mpd, msd
        
        
def main():
    mel_config = MelSpectrogramConfig()
    model_config = ModelConfig()
    train_config = TrainConfig()
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(train_config.seed)
    torch.cuda.manual_seed(train_config.seed)
    
    device = train_config.device

    train_dataset = MelDataset(train_config.train_wav_path, train_config.segment_size)
    test_dataset = MelDataset(train_config.test_wav_path, -1)

    train_loader = DataLoader(train_dataset, num_workers=train_config.num_workers, shuffle=True, batch_size=train_config.batch_size, pin_memory=True, drop_last=True)
    
    generator = Generator(model_config).to(device)
    
    mpd = MPDiscriminator(model_config).to(device)
    
    
    msd = MSDiscriminator().to(device)
    
    
    optim_g = torch.optim.AdamW(generator.parameters(), lr=train_config.learning_rate, betas=(train_config.adam_b1, train_config.adam_b1))
    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()), lr=train_config.learning_rate, betas=(train_config.adam_b1, train_config.adam_b1))

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=train_config.gamma)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=train_config.gamma)

    meltransform = MelSpectrogram(mel_config).to(device)
    
    if not os.path.exists(train_config.checkpoint_dir):
        os.mkdir(train_config.checkpoint_dir)
    if os.path.exists(train_config.checkpoint_path):
        generator, mpd, msd = load_checkpoint(train_config.checkpoint_path, generator, mpd, msd)
        print('loaded')
    
    generator.train()
    mpd.train()
    msd.train()
    
    last_epoch = train_config.last_epoch
    current_step = (last_epoch + 1) * len(train_loader)
    
    logger = WanDBWriter(train_config.wandb_project)
    tqdm_bar = tqdm(total=(train_config.num_epochs - last_epoch - 1) * len(train_loader))

    for epoch in range(last_epoch + 1, train_config.num_epochs):
        for idx, batch in enumerate(train_loader):
            current_step += 1
            logger.set_step(current_step)
            
            tqdm_bar.update(1)
        
            wav = batch.to(device)
            mel = meltransform(wav)
            wav = wav.unsqueeze(1)
        
            fake_wav = generator(mel)
            fake_mel = meltransform(fake_wav)

            optim_d.zero_grad()
            d_reals, d_fakes, _, _ = mpd(wav, fake_wav.detach())
            mpd_loss = discriminator_loss(d_reals, d_fakes)
        
            d_reals, d_fakes, _, _ = msd(wav, fake_wav.detach())
            msd_loss = discriminator_loss(d_reals, d_fakes)
            loss_disc = msd_loss + mpd_loss
            loss_disc.backward()
            optim_d.step()

            optim_g.zero_grad()
            d_reals, d_fakes, fmap_reals, fmap_fakes = mpd(wav, fake_wav)
            feature_loss_mpd = feature_loss(fmap_reals, fmap_fakes) * train_config.feature_mul
            gen_loss_mpd = generator_loss(d_fakes)
        
            d_reals, d_fakes, fmap_reals, fmap_fakes = msd(wav, fake_wav)
            feature_loss_msd = feature_loss(fmap_reals, fmap_fakes) * train_config.feature_mul
            gen_loss_msd = generator_loss(d_fakes)
        
            loss_mel = mel_loss(mel, fake_mel) * train_config.mel_mul
            gen_loss = gen_loss_mpd + gen_loss_msd + feature_loss_msd + feature_loss_mpd + loss_mel
            gen_loss.backward()
            optim_g.step()
            

            if current_step % train_config.log_step == 0 and current_step != 0:
                logger.add_scalar('mpd_loss', mpd_loss)
                logger.add_scalar('msd_loss', msd_loss)
                logger.add_scalar('discriminator_loss', loss_disc)
                logger.add_scalar('gen_loss_mpd', gen_loss_mpd)
                logger.add_scalar('gen_loss_msd', gen_loss_msd)
                logger.add_scalar('feature_loss_mpd', feature_loss_mpd)
                logger.add_scalar('feature_loss_msd', feature_loss_msd)
                logger.add_scalar('mel_loss', loss_mel)
                logger.add_scalar('generator_loss', gen_loss)
            
                logger.set_step(current_step, 'test')
                generator.eval()
                torch.cuda.empty_cache()
                
                with torch.no_grad():
                    mel_error  = 0.
                    for j, wav in enumerate(test_dataset):
                        wav = wav.unsqueeze(0).to(device)
                        mel = meltransform(wav)
                        
                        fake_wav = generator(mel).squeeze(1)
                        
                        fake_mel = meltransform(fake_wav)

                        logger.add_audio('audio_'+str(j+1), fake_wav.squeeze(), mel_config.sr)
                        
                        mel_error += mel_loss(mel, fake_mel).item()

                    logger.add_scalar('mel_loss', mel_error)
                generator.train()
            
            if current_step % train_config.freq_save_model == 0 and current_step != 0:
                save_checkpoint(train_config.checkpoint_path + str(current_step), generator, mpd, msd)
        
        save_checkpoint(train_config.checkpoint_path, generator, mpd, msd)

        scheduler_g.step()
        scheduler_d.step()

if __name__ == '__main__':
    main()