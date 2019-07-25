import numpy as np
#from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

from utils import get_last_half, enumerate_discrete_latents
from modules import ConvBlock, TransposedConvBlock, VQEmbedding
from functions import gumbel_softmax

class AutoEncoder(nn.Module):
    def __init__(self,
                 data_distribution = 'gaussian',
                 input_dim = 360,
                 z_dim = 30,
                 kernel = 4,
                 stride = 2,
                 padding = 1):
        super(AutoEncoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = z_dim

        n, last_half = get_last_half(input_dim)
        n = min(int( np.log(input_dim/1.0)/np.log(2) ), n)
        final_kernel = int( input_dim / (2**n) )

        self.encoder = nn.Sequential(
                            ConvBlock(1, z_dim, kernel, stride, padding))
        self.decoder = nn.Sequential(
                            TransposedConvBlock(z_dim, z_dim, final_kernel, 1, 0))
        for i in range(n - 1):
            self.encoder.add_module("conv"+str(i+1), ConvBlock(z_dim, z_dim, kernel, stride, padding))
            self.decoder.add_module("Tconv"+str(i+1), TransposedConvBlock(z_dim, z_dim, kernel, stride, padding))
        self.encoder.add_module("z_conv", ConvBlock(z_dim, z_dim, final_kernel, 1, 0))
        self.decoder.add_module("x_Tconv", nn.ConvTranspose1d(z_dim, 1, kernel, stride, padding))

        if data_distribution == 'bernoulli':
            self.decoder.add_module("sigmoid", nn.Sigmoid())

    def encode(self, x):
        z = self.encoder(x).permute(0, 2, 1)
        return z

    def decode(self, z):
        x_hat = self.decoder(z.permute(0, 2, 1))
        return x_hat

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat

class VAE(AutoEncoder):
    def __init__(self,
                 data_distribution = 'gaussian',
                 input_dim = 360,
                 z_dim = 30,
                 kernel = 4,
                 stride = 2,
                 padding = 1):
        super(VAE, self).__init__(data_distribution,
                                  input_dim,
                                  z_dim,
                                  kernel,
                                  stride,
                                  padding)

        # n, last_half = get_last_half(input_dim)
        # n = min( int( np.log(input_dim/z_dim)/np.log(2) ), n)
        # final_kernel = int( input_dim / (2**n) )

        n, last_half = get_last_half(input_dim)
        n = min(int( np.log(input_dim/1.0)/np.log(2) ), n)
        final_kernel = int( input_dim / (2**n) )

        # last_module_removed = list(self.encoder.children())[:-1]
        # self.encoder = torch.nn.Sequential(*last_module_removed)
        self.encoder = self.encoder[:-1]
        self.encoder.add_module("z_conv", ConvBlock(z_dim, 2*z_dim, final_kernel, 1, 0))

        if data_distribution == 'bernoulli':
            self.decoder.add_module("sigmoid", nn.Sigmoid())

    def encode(self, x):
        h = super(VAE, self).encode(x)
        mu, logvar = h.chunk(2, dim=2)
        distribution_params = (mu, logvar)
        return distribution_params

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            z = mu + eps*std
        else:
            z = mu
        return z

    def forward(self, x):
        distribution_params = self.encode(x)
        mu, logvar = distribution_params
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# class DiscreteLatentVAE1(VAE):
#     def __init__(self,
#                  data_distribution = 'gaussian',
#                  input_dim = 360,
#                  latent_dim = 6,
#                  categorical_dim = 5,
#                  kernel = 4,
#                  stride = 2,
#                  padding = 1):
#         super(DiscreteLatentVAE1, self).__init__(data_distribution,
#                                                 input_dim,
#                                                 latent_dim * categorical_dim,
#                                                 kernel,
#                                                 stride,
#                                                 padding)
#
#         self.latent_dim = latent_dim
#         self.categorical_dim = categorical_dim
#
#         dim = latent_dim * categorical_dim
#         n, last_half = get_last_half(input_dim)
#         n = min( int( np.log(input_dim/dim)/np.log(2) ), n)
#         final_kernel = int( input_dim / (2**n) )
#
#         last_module_removed = list(self.encoder.children())[:-1]
#         self.encoder = torch.nn.Sequential(*last_module_removed)
#         self.encoder.add_module("z_conv", ConvBlock(dim, dim, final_kernel, 1, 0))
#
#         if data_distribution == 'bernoulli':
#             self.decoder.add_module("sigmoid", nn.Sigmoid())
#
#     def encode(self, x):
#         q = self.encoder(x).permute(0, 2, 1)
#         z = q.view(q.size(0), self.latent_dim, self.categorical_dim)
#         return z
#
#     def decode(self, z):
#         z = z.view(z.size(0), 1, self.latent_dim * self.categorical_dim)
#         x_hat = super(DiscreteLatentVAE1, self).decode(z)
#         return x_hat

class DiscreteLatentVAE(VAE):
    def __init__(self,
                 data_distribution = 'gaussian',
                 input_dim = 360,
                 latent_dim = 10,
                 n_latents = 6,
                 categorical_dim = 5,
                 kernel = 4,
                 stride = 2,
                 padding = 1):
        super(DiscreteLatentVAE, self).__init__(data_distribution,
                                                input_dim,
                                                latent_dim,
                                                kernel,
                                                stride,
                                                padding)

        self.latent_dim = latent_dim
        self.n_latents = n_latents
        self.categorical_dim = categorical_dim

        n, last_half = get_last_half(input_dim)
        n = min(int( np.log(input_dim/n_latents)/np.log(2) ), n)
        final_conv_size = int( input_dim / (2**n) )
        final_kernel = final_conv_size - (n_latents - 1)

        if final_kernel <= 0:
            raise ValueError('n_latents have to be lower than: '+str(final_conv_size/stride + 1))

        self.encoder = nn.Sequential(
                            ConvBlock(1, latent_dim, kernel, stride, padding))
        self.decoder = nn.Sequential(
                            TransposedConvBlock(latent_dim, latent_dim, final_kernel, 1, 0))
        for i in range(n - 1):
            self.encoder.add_module("conv"+str(i+1), ConvBlock(latent_dim, latent_dim, kernel, stride, padding))
            self.decoder.add_module("Tconv"+str(i+1), TransposedConvBlock(latent_dim, latent_dim, kernel, stride, padding))
        self.encoder.add_module("z_conv", ConvBlock(latent_dim, latent_dim, final_kernel, 1, 0))
        self.decoder.add_module("x_Tconv", nn.ConvTranspose1d(latent_dim, 1, kernel, stride, padding))

        if data_distribution == 'bernoulli':
            self.decoder.add_module("sigmoid", nn.Sigmoid())

    def encode(self, x):
        q = self.encoder(x).permute(0, 2, 1)
        z = q.view(q.size(0), self.latent_dim, self.categorical_dim)
        return z

    def decode(self, z):
        z = z.view(z.size(0), 1, self.latent_dim * self.categorical_dim)
        x_hat = super(DiscreteLatentVAE, self).decode(z)
        return x_hat


class GumbelVAE(DiscreteLatentVAE):
    def __init__(self,
                 data_distribution = 'gaussian',
                 input_dim = 360,
                 latent_dim = 6,
                 categorical_dim = 5,
                 kernel = 4,
                 stride = 2,
                 padding = 1):
        super(GumbelVAE, self).__init__(data_distribution,
                                        input_dim,
                                        latent_dim = latent_dim*categorical_dim,
                                        n_latents = 1,
                                        categorical_dim = 5,
                                        kernel = kernel,
                                        stride = stride,
                                        padding = padding)
        self.latent_dim = latent_dim
        self.categorical_dim = categorical_dim

        # dim = latent_dim * categorical_dim
        # n, last_half = get_last_half(input_dim)
        # n = min( int( np.log(input_dim/dim)/np.log(2) ), n)
        # final_kernel = int( input_dim / (2**n) )
        #
        # last_module_removed = list(self.encoder.children())[:-1]
        # self.encoder = torch.nn.Sequential(*last_module_removed)
        # self.encoder.add_module("z_conv", ConvBlock(dim, dim, final_kernel, 1, 0))

        if data_distribution == 'bernoulli':
            self.decoder.add_module("sigmoid", nn.Sigmoid())

    def encode(self, x,
               temp=0.5, hard=True,
               discrete_labels=False, enumerate_labels=False):
        q_y = super(GumbelVAE, self).encode(x)
        z = gumbel_softmax(q_y, temp, hard)
        if enumerate_labels:
            q_y = torch.argmax(q_y.view(q_y.size(0), self.latent_dim, self.categorical_dim), dim = 2)
            enumeration = enumerate_discrete_latents(q_y, self.categorical_dim)
            return enumeration
        if discrete_labels:
            q_y = torch.argmax(q_y.view(q_y.size(0), self.latent_dim, self.categorical_dim), dim = 2)
            return q_y
        return z, q_y

    def forward(self, x, temp=0.5, hard=False):
        z, q_y = self.encode(x, temp, hard)
        x_hat = self.decode(z)
        return x_hat, F.softmax(q_y, dim=-1).reshape((q_y.size(0), 1, self.latent_dim * self.categorical_dim))


# class GumbelVAE(DiscreteLatentVAE1):
#     def __init__(self,
#                  data_distribution = 'gaussian',
#                  input_dim = 360,
#                  latent_dim = 6,
#                  categorical_dim = 5,
#                  kernel = 4,
#                  stride = 2,
#                  padding = 1):
#         super(GumbelVAE, self).__init__(data_distribution,
#                                         input_dim,
#                                         latent_dim,
#                                         categorical_dim,
#                                         kernel,
#                                         stride,
#                                         padding)
#
#     def encode(self, x, temp=0.5, hard=True):
#         q_y = super(GumbelVAE, self).encode(x)
#         z = gumbel_softmax(q_y, temp, hard)
#         return z, q_y
#
#     def forward(self, x, temp=0.5, hard=False):
#         z, q_y = self.encode(x, temp, hard)
#         x_hat = self.decode(z)
#         return x_hat, F.softmax(q_y, dim=-1).reshape((q_y.size(0), 1, self.latent_dim * self.categorical_dim))


class VectorQuantizedVAE(DiscreteLatentVAE):
    def __init__(self,
                 data_distribution = 'gaussian',
                 input_dim = 360,
                 latent_dim = 10,
                 n_latents = 6,
                 categorical_dim = 5,
                 kernel = 4,
                 stride = 2,
                 padding = 1):
        super(VectorQuantizedVAE, self).__init__(data_distribution,
                                                input_dim,
                                                latent_dim,
                                                n_latents,
                                                categorical_dim,
                                                kernel,
                                                stride,
                                                padding)
        self.dim = self.latent_dim
        self.n_latents = n_latents
        self.K = self.categorical_dim
        self.codebook = VQEmbedding(self.K, self.dim)

    def encode(self, x, enumerate_labels=False):
        z_e_x = self.encoder(x)
        z = self.codebook(z_e_x)
        if enumerate_labels:
            enumeration = enumerate_discrete_latents(z, self.categorical_dim)
            return enumeration
        return z

    def decode(self, z):
        z_q_x = self.codebook.embedding(z).permute(0, 2, 1).contiguous()
        x_tilde = self.decoder(z_q_x)
        return x_tilde

    def forward(self, x):
        z_e_x = self.encoder(x)
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        x_tilde = self.decoder(z_q_x_st)
        return x_tilde, z_e_x, z_q_x
