import torch

from torch import nn

def vae_loss(recon_x, x, mean, log_var):
    """
    This function will add the reconstruction loss (MSE loss) and the 
    KL-Divergence.
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param BCE: binary cross entropy loss
    :param mu: the mean from the latent vector
    :param logvar: log variance from the latent vector
    """
    BCE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = _kld_loss(mean, log_var)
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
    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes):

        super().__init__()

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size
        self.encoder = Encoder(encoder_layer_sizes, latent_size)
        self.decoder = Decoder(decoder_layer_sizes, latent_size)

    def forward(self, x):
        means, log_var = self.encoder(x)
        z = reparameterize(means, log_var)
        recon_x = self.decoder(z)

        return recon_x, means, log_var

    def inference(self, z):
        return self.decoder(z)
    
class Encoder(nn.Module):
    """ Class for encoder """
    def __init__(self, layer_sizes, latent_size):

        super().__init__()

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x):
        x = self.MLP(x)
    
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class Decoder(nn.Module):
    """ Class for decoder """
    def __init__(self, layer_sizes, latent_size):

        super().__init__()

        self.MLP = nn.Sequential()

        input_size = latent_size

        for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            else:
                self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())

    def forward(self, z):
        x = self.MLP(z)
        return x