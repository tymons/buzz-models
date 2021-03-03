from torch import nn
from utils.pytorch_impl.vae import Encoder, Decoder, reparameterize

class cVAE(nn.Module):
    """ Class for Contrastive autoencoder """
    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes):

        super().__init__()

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size
        self.s_encoder = Encoder(encoder_layer_sizes, latent_size)
        self.z_encoder = Encoder(encoder_layer_sizes, latent_size)
        self.decoder = Decoder(decoder_layer_sizes, 2 * latent_size)

    def forward(self, target, background):
        tg_s_mean, tg_s_log_var = self.encoder_s(target)
        tg_z_mean, tg_z_log_var = self.encoder_z(target)
        bg_z_mean, bg_z_log_var = self.encoder_z(background)    
    
        tg_s = self.reparameterize(tg_s_mean, tg_s_log_var)
        tg_z = self.reparameterize(tg_z_mean, tg_z_log_var)
        bg_z = self.reparameterize(bg_z_mean, bg_z_log_var)

        tg_output = self.decoder(torch.cat((tg_z, tg_s), axis=2))
        bg_output = self.decoder(torch.cat((bg_z, torch.zeros_like(tg_s)), axis=2))

        return (tg_output, tg_s_mean, tg_s_log_var, tg_z_mean, tg_z_log_var), (bg_output, bg_z_mean, bg_z_log_var)