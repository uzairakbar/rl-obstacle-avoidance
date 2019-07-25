import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel = 4,
                 stride = 2,
                 padding = 1):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel, stride, padding),
            nn.BatchNorm1d(out_channel),
            nn.LeakyReLU(inplace=True)
            # nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)


class TransposedConvBlock(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel = 4,
                 stride = 2,
                 padding = 1):
        super(TransposedConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose1d(in_channel, out_channel, kernel, stride, padding),
            nn.BatchNorm1d(out_channel),
            nn.LeakyReLU(inplace=True)
            # nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)


class Interpolate(nn.Module):
    def __init__(self, size, mode='linear'):
        super(Interpolate, self).__init__()
        self.interp = F.interpolate
        self.size = size
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x


class ResBlock(nn.Module):
    def __init__(self, dim = 30):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(dim, dim, 3, 1, 1),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim, dim, 1),
            nn.BatchNorm1d(dim)
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(x + self.block(x))


from functions import VectorQuantization, VectorQuantizationStraightThrough

# based on implementation of Rithesh Kumar et al :
# https://github.com/ritheshkumar95/pytorch-vqvae
vq, vq_st = VectorQuantization.apply, VectorQuantizationStraightThrough.apply
class VQEmbedding(nn.Module):
    def __init__(self, K, D):
        super(VQEmbedding, self).__init__()
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1./K, 1./K)

    def forward(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 1).contiguous()
        latents = vq(z_e_x_, self.embedding.weight)
        return latents

    def straight_through(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 1).contiguous()
        z_q_x_, indices = vq_st(z_e_x_, self.embedding.weight.detach())
        z_q_x = z_q_x_.permute(0, 2, 1).contiguous()

        z_q_x_bar_flatten = torch.index_select(self.embedding.weight,
            dim=0, index=indices)
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        z_q_x_bar = z_q_x_bar_.permute(0, 2, 1).contiguous()

        return z_q_x, z_q_x_bar
