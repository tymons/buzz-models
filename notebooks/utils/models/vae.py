import torch
import torch.nn.functional as F

from tqdm import tqdm
from torch import nn
from utils.data_utils import batch_normalize, batch_standarize
from utils.models.ae import Encoder, Decoder


def vae_loss(data_input, target, mean, logvar):
    """
    This function will add the reconstruction loss (MSE loss) and the 
    KL-Divergence.
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param BCE: binary cross entropy loss
    :param mu: the mean from the latent vector
    :param logvar: log variance from the latent vector
    """
    BCE = F.mse_loss(data_input, target, reduction='sum')
    KLD = _kld_loss(mean, logvar)
    return (BCE + KLD)

def _kld_loss(mean, log_var):
    """ KLD loss for normal distribution"""
    return -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    
def reparameterize(mu, log_var):
    """ Function for reparametrization trick """
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)

    return mu + eps * std

class VAE(nn.Module):
    """ Class for variational autoencoder """
    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes, input_size):

        super().__init__()

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list
        assert type(input_size) == int

        self.latent_size = latent_size
        self.encoder = Encoder(encoder_layer_sizes, input_size)
        self.decoder = Decoder(decoder_layer_sizes, latent_size, input_size)
        
        self.linear_means = nn.Linear(encoder_layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(encoder_layer_sizes[-1], latent_size)

    def forward(self, x):
        y = self.encoder(x)
        means = self.linear_means(y)
        log_var =  self.linear_log_var(y)
        z = reparameterize(means, log_var)
        recon_x = self.decoder(z)

        return {'target': recon_x, 'mean': means, 'logvar': log_var}
    