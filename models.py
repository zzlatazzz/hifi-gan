import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

from configs import ModelConfig

config = ModelConfig()
LRELU_SLOPE = config.leaky_relu_slope

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)
        
def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)

    
class ResBlock(torch.nn.Module):
    def __init__(self, channels, k_r_i, d_r_i):
        super(ResBlock, self).__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, k_r_i, 1, dilation=d_r_i[i][0],
                               padding=get_padding(k_r_i, d_r_i[i][0]))) for i in range(len(d_r_i))])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, k_r_i, 1, dilation=d_r_i[i][1],
                               padding=get_padding(k_r_i, d_r_i[i][1]))) for i in range(len(d_r_i))])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for conv1, conv2 in zip(self.convs1, self.convs2):
            xt = conv2(nn.LeakyReLU(LRELU_SLOPE)(conv1(nn.LeakyReLU(LRELU_SLOPE)(x))))
            x = xt + x
        return x


class MRF(torch.nn.Module):
    def __init__(self, channels, k_r, D_r):
        super(MRF, self).__init__()
        
        self.net = nn.ModuleList([ResBlock(channels, k_r[i], D_r) for i in range(len(k_r))])
        
    def forward(self, x):
        out = self.net[0](x)
        for i in range(1, len(self.net)):
            out = out + self.net[i](x)
        return out / len(self.net)
    
        
class Generator(torch.nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        
        h_u, k_u, k_r, D_r = config.upsample_initial_channel, config.upsample_kernel_sizes, config.resblock_kernel_sizes, config.resblock_dilation_sizes
        
        self.first_conv = nn.utils.weight_norm(nn.Conv1d(80, h_u, 7, 1, padding=3))
        self.net = nn.ModuleList([nn.Sequential(nn.LeakyReLU(),
                                                nn.utils.weight_norm(nn.ConvTranspose1d(h_u // (2 ** i), h_u // (2 ** (i + 1)), k_u[i], stride=k_u[i] // 2, padding=(k_u[i] - k_u[i] // 2)// 2)),
                                                MRF(h_u // (2 ** (i + 1)), k_r, D_r)) for i in range(4)])
        self.last_block = nn.Sequential(nn.LeakyReLU(), nn.utils.weight_norm(nn.Conv1d(h_u // (2 ** 4), 1, 7, 1, padding=3)), nn.Tanh())
        
    def forward(self, x):
        x = self.first_conv(x)
        for i in range(len(self.net)):
            x = self.net[i](x)
        x = self.last_block(x)
        return x


class SubMPDiscriminator(torch.nn.Module):
    def __init__(self, p):
        super(SubMPDiscriminator, self).__init__()
        self.p = p
        
        self.net = nn.ModuleList([ nn.Sequential(
            nn.utils.weight_norm(nn.Conv2d(1 if not i else 2 ** (5 + (i - 1) * 2), 2 ** (5 + i * 2) if i != 3 else 1024, (5, 1), stride=(3, 1), padding=(get_padding(5), 0))),
            nn.LeakyReLU()
        ) for i in range(4)] + [nn.Sequential(
            nn.utils.weight_norm(nn.Conv2d(1024, 1024, (5, 1), stride=1, padding=(2, 0))),
            nn.LeakyReLU()),
            nn.utils.weight_norm(nn.Conv2d(1024, 1, (3, 1), stride=1, padding=(1, 0)))])
            
    def forward(self, x):
        fmap = []
        x = F.pad(x, (0, self.p - (x.shape[2] % self.p)), "reflect")
        x = x.view(x.shape[0], x.shape[1], x.shape[2] // self.p, self.p)
        for i in range(len(self.net)):
            x = self.net[i](x)
            fmap.append(x)
        x = nn.Flatten()(x)
        return x, fmap


class MPDiscriminator(torch.nn.Module):
    def __init__(self, config):
        super(MPDiscriminator, self).__init__()
        
        periods = config.periods
        self.discriminators = nn.ModuleList([SubMPDiscriminator(p) for p in periods])

    def forward(self, y, y_hat):
        
        d_reals, d_fakes, fmap_reals, fmap_fakes = [], [], [], []
        for d in self.discriminators:
            d_real, fmap_real = d(y)
            d_fake, fmap_fake = d(y_hat)
            d_reals.append(d_real)
            d_fakes.append(d_fake)
            fmap_reals.append(fmap_real)
            fmap_fakes.append(fmap_fake)
        return d_reals, d_fakes, fmap_reals, fmap_fakes


class SubMSDiscriminator(torch.nn.Module):
    def __init__(self, norm):
        super(SubMSDiscriminator, self).__init__()
        self.net = nn.ModuleList([
            nn.Sequential(norm(nn.Conv1d(1, 128, 15, 1, padding=7)), nn.LeakyReLU()),
            nn.Sequential(norm(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)), nn.LeakyReLU()),
            nn.Sequential(norm(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)), nn.LeakyReLU()),
            nn.Sequential(norm(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)), nn.LeakyReLU()),
            nn.Sequential(norm(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)), nn.LeakyReLU()),
            nn.Sequential(norm(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)), nn.LeakyReLU()),
            nn.Sequential(norm(nn.Conv1d(1024, 1024, 5, 1, padding=2)), nn.LeakyReLU()),
            norm(nn.Conv1d(1024, 1, 3, 1, padding=1))
        ])

    def forward(self, x):
        fmap = []
        for i in range(len(self.net)):
            x = self.net[i](x)
            fmap.append(x)
        x = nn.Flatten()(x)
        return x, fmap


class MSDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MSDiscriminator, self).__init__()
        
        self.discriminators = nn.ModuleList([
            SubMSDiscriminator(nn.utils.spectral_norm),
            nn.Sequential(nn.AvgPool1d(4, 2, padding=2), SubMSDiscriminator(nn.utils.weight_norm)),
            nn.Sequential(nn.AvgPool1d(4, 2, padding=2), SubMSDiscriminator(nn.utils.weight_norm))])

    def forward(self, y, y_hat):
        d_reals, d_fakes, fmap_reals, fmap_fakes = [], [], [], []
        for d in self.discriminators:
            d_real, fmap_real = d(y)
            d_fake, fmap_fake = d(y_hat)
            d_reals.append(d_real)
            d_fakes.append(d_fake)
            fmap_reals.append(fmap_real)
            fmap_fakes.append(fmap_fake)
        return d_reals, d_fakes, fmap_reals, fmap_fakes