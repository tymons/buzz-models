import torch
import math
import numpy as np

from torch import nn
from utils.models.vae import reparameterize
from utils.models.conv_ae import ConvolutionalDecoder, ConvolutionalEncoder


class ConvolutionalVAE(nn.Module):
    """ Class for convolutional variational autoencoder """
    def __init__(self, encoder_conv_sizes, encoder_mlp_sizes,
                        decoder_conv_sizes, decoder_mlp_sizes, latent_size, input_size):

        assert type(encoder_conv_sizes) == list
        assert type(encoder_mlp_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_conv_sizes) == list
        assert type(decoder_mlp_sizes) == list
        assert type(input_size) == tuple   

        super().__init__()
        
        self.encoder = ConvolutionalEncoder(encoder_conv_sizes, encoder_mlp_sizes)
        self.decoder = ConvolutionalDecoder(decoder_conv_sizes, decoder_mlp_sizes, latent_size, (input_size[0]//input_size[1]))
        self.linear_means = nn.Linear(encoder_mlp_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(encoder_mlp_sizes[-1], latent_size)
        self.latent_size = latent_size

    def forward(self, x):
        y = self.encoder(x)
        means = self.linear_means(y)
        log_var = self.linear_log_var(y)
        latent = reparameterize(means, log_var)
        recon_x = self.decoder(latent)

        return {'target': recon_x, 'mean': means, 'logvar': log_var}