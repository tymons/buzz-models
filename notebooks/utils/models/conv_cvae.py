import torch

from torch import nn

from utils.models.conv_vae import ConvolutionalDecoder, ConvolutionalEncoder
from utils.models.vae import reparameterize

class ConvolutionalCVAE(nn.Module):
    """ Class for convolutional cvae """
    def __init__(self, encoder_conv_sizes, encoder_mlp_sizes,
                    decoder_conv_sizes, decoder_mlp_sizes, latent_size, input_size):

        super().__init__()	

        assert type(encoder_conv_sizes) == list
        assert type(encoder_mlp_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_conv_sizes) == list
        assert type(decoder_mlp_sizes) == list

        self.latent_size = latent_size
        self.s_encoder = ConvolutionalEncoder(encoder_conv_sizes, encoder_mlp_sizes, latent_size)
        self.z_encoder = ConvolutionalEncoder(encoder_conv_sizes, encoder_mlp_sizes, latent_size)
        self.decoder = ConvolutionalDecoder(decoder_conv_sizes, decoder_mlp_sizes, 2 * latent_size, input_size[0]//input_size[1])

    def forward(self, target, background):
        tg_s_mean, tg_s_log_var = self.s_encoder(target)
        tg_z_mean, tg_z_log_var = self.z_encoder(target)
        bg_z_mean, bg_z_log_var = self.z_encoder(background)    
    
        tg_s = reparameterize(tg_s_mean, tg_s_log_var)
        tg_z = reparameterize(tg_z_mean, tg_z_log_var)
        bg_z = reparameterize(bg_z_mean, bg_z_log_var)

        tg_output = self.decoder(torch.cat((tg_z, tg_s), axis=1))
        bg_output = self.decoder(torch.cat((bg_z, torch.zeros_like(tg_s)), axis=1))

        return {'target': tg_output,
                'tg_qs_mean': tg_s_mean,
                'tg_qs_log_var': tg_s_log_var,
                'tg_qz_mean': tg_z_mean,
                'tg_qz_log_var': tg_z_log_var,
                'background': bg_output,
                'bg_qz_mean': bg_z_mean,
                'bg_qz_log_var': bg_z_log_var,
                'latent_qs_target': tg_s,       # we need this for disentangle and ensure that s and z distributions 
                'latent_qz_target': tg_z}       # for target are independent

