import torch
from torch import nn

def generator_loss(outs):
    loss = 0
    for out in outs:
        loss += torch.mean((1 - out)**2)
    return loss

def discriminator_loss(real_outs, fake_outs):
    loss = 0
    for real_out, fake_out in zip(real_outs, fake_outs):
        loss += torch.mean((1 - real_out) ** 2) + torch.mean(fake_out ** 2)
    return loss

def feature_loss(real_fmaps, fake_fmaps):
    loss = 0
    for real_fmap, fake_fmap in zip(real_fmaps, fake_fmaps):
        for r_fmap, f_fmap in zip(real_fmap, fake_fmap):
            loss += (r_fmap - f_fmap).abs().mean()
    return loss

def mel_loss(real_melspec, fake_melspec):
    return (real_melspec - fake_melspec).abs().mean()