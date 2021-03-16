import math
import torch

from torch import nn


def _conv2d_block(in_f, out_f, *args, **kwargs):
    """ Function for building convolutional block

        Attributes
            in_f - number of input features
            out_f - number of output features
    """
    return nn.Sequential(
        nn.Conv2d(in_f, out_f, *args, **kwargs),
        nn.BatchNorm2d(out_f),
        nn.ReLU(),
        nn.Dropout2d(p=0.2)
    )

def _conv2d_transpose_block(in_f, out_f, *args, **kwargs):
    """ Function for building transpose convolutional block

        Attributes
            in_f - number of input features
            out_f - number of output features
    """
    return nn.Sequential(
        nn.ConvTranspose2d(in_f, out_f, *args, **kwargs),
        nn.BatchNorm2d(out_f),
        nn.ReLU(),
        nn.Dropout2d(p=0.2)
    )

class View(nn.Module):
    """ Function for nn.Sequentional to reshape data """

    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class ConvolutionalAE(nn.Module):
    """ Class for convolutional autoencoder """
    def __init__(self, encoder_conv_sizes, encoder_mlp_sizes,
                    decoder_conv_sizes, decoder_mlp_sizes, latent_size, input_size):
            
        assert type(encoder_conv_sizes) == list
        assert type(encoder_mlp_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_conv_sizes) == list
        assert type(decoder_mlp_sizes) == list
        assert type(input_size) == tuple   

        super().__init__()     

        self.encoder = ConvolutionalEncoderWithLatent(encoder_conv_sizes, encoder_mlp_sizes, latent_size)
        self.decoder = ConvolutionalDecoder(decoder_conv_sizes, decoder_mlp_sizes, latent_size, (input_size[0]//input_size[1]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class ConvolutionalEncoder(nn.Module):
    """ Class for conv encoder without last latent layer as this class should be used in both vae and ae """
    def __init__(self, conv_features_sizes, linear_layer_sizes):
        super().__init__()
        
        self.conv = nn.Sequential()
        self.mlp = nn.Sequential()
        self.flat = nn.Flatten()

        self.conv.add_module(name=f"e-fconv{0}", module=_conv2d_block(1, conv_features_sizes[0], kernel_size=3, padding=1))
        self.conv.add_module(name=f"e-max{0}", module=nn.MaxPool2d(2, 2))
        for i, (in_size, out_size) in enumerate(zip(conv_features_sizes[:-1], conv_features_sizes[1:]), 1):
            self.conv.add_module(name=f"e-fconv{i}", module=_conv2d_block(in_size, out_size, kernel_size=3, padding=1))
            self.conv.add_module(name=f"e-max{i}", module=nn.MaxPool2d(2, 2))

        for i, (in_size, out_size) in enumerate(zip(linear_layer_sizes[:-1], linear_layer_sizes[1:])):
            self.mlp.add_module(name=f"e-linear{i}", module=nn.Linear(in_size, out_size))
            self.mlp.add_module(name=f"e-batchnorm{i}", module=nn.BatchNorm1d(out_size))
            self.mlp.add_module(name=f"e-relu{i}", module=nn.ReLU())

    def forward(self, x):
        x = self.conv(x)
        x = self.flat(x)
        x = self.mlp(x)
        return x
        
class ConvolutionalDecoder(nn.Module):
    """ Class for conv decoder """
    def __init__(self, conv_features_sizes, linear_sizes, latent_size, input_ratio):
        super().__init__()

        self.conv = nn.Sequential()
        self.mlp = nn.Sequential()
        self.sigmoid = nn.Sigmoid()

        self.mlp.add_module(name=f"d-linear{0}", module=nn.Linear(latent_size, linear_sizes[0]))
        for i, (in_size, out_size) in enumerate(zip(linear_sizes[:-1], linear_sizes[1:]), 1):
            self.mlp.add_module(name=f"d-linear{i}", module=nn.Linear(in_size, out_size))
            self.mlp.add_module(name=f"d-batchnorm{i}", module=nn.BatchNorm1d(out_size))
            self.mlp.add_module(name=f"d-relu{i}", module=nn.ReLU())
    
        w_h = int(math.sqrt(linear_sizes[-1]//conv_features_sizes[0]))      # width and height
        input_ratio = int(math.sqrt(input_ratio))
        w = input_ratio*w_h
        h = w_h//input_ratio
        desired_shape = torch.reshape(torch.empty(linear_sizes[-1]), (conv_features_sizes[0], w, h)).shape
        self.view = View([-1, *desired_shape])

        for i, (in_size, out_size) in enumerate(zip(conv_features_sizes[:-1], conv_features_sizes[1:])):
            self.conv.add_module(name=f"d-fconv{i}", module=_conv2d_transpose_block(in_size, out_size, kernel_size=2, stride=2))
        self.conv.add_module(name=f"d-conv{i}", module=nn.ConvTranspose2d(conv_features_sizes[-1], 1, kernel_size=2, stride=2))
        

    def forward(self, x):
        x = self.mlp(x)
        x = self.view(x)
        x = self.conv(x)
        x = self.sigmoid(x)
        return x

class ConvolutionalEncoderWithLatent(ConvolutionalEncoder):
    """ Class for convolutional encoder with last layer for latent vector """
    def __init__(self, conv_features_sizes, linear_layer_sizes, latent_size):
        super().__init__(conv_features_sizes, linear_layer_sizes)
        # add last fc layer to get latent vector
        self.latent_layer = nn.Linear(linear_layer_sizes[-1], latent_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.flat(x)
        x = self.mlp(x)
        x = self.latent_layer(x)
        x = self.relu(x)
        return x