import torch 
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from tqdm import tqdm

def _reparameterize(mu, log_var):
    """ Function for reparametrization trick """
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)

    return mu + eps * std

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

class ConvolutionalVAE(nn.Module):
    """ Class for convolutional variational autoencoder """
    def __init__(self, encoder_conv_sizes, encoder_mlp_sizes,
                        decoder_conv_sizes, decoder_mlp_sizes, latent):
        super().__init__()
        self.encoder = ConvolutionalEncoder(encoder_conv_sizes, encoder_mlp_sizes, latent)
        self.decoder = ConvolutionalDecoder(decoder_conv_sizes, decoder_mlp_sizes, latent)

    def forward(self, x):
        means, log_var = self.encoder(x)
        latent = _reparameterize(means, log_var)
        recon_x = self.decoder(latent)

        return recon_x, means, log_var

class ConvolutionalDecoder(nn.Module):
    """ Class for conv decoder """
    def __init__(self, conv_features_sizes, linear_sizes, latent):
        super().__init__()

        self.conv = nn.Sequential()
        self.mlp = nn.Sequential()
        self.sigmoid = nn.Sigmoid()

        self.mlp.add_module(name=f"d-linear{0}", module=nn.Linear(latent, linear_sizes[0]))
        for i, (in_size, out_size) in enumerate(zip(linear_sizes[:-1], linear_sizes[1:]), 1):
            self.mlp.add_module(name=f"d-linear{i}", module=nn.Linear(in_size, out_size))
            self.mlp.add_module(name=f"d-batchnorm{i}", module=nn.BatchNorm1d(out_size))
            self.mlp.add_module(name=f"d-relu{i}", module=nn.ReLU())
    
        temp_tensor = torch.empty(linear_sizes[-1])
        divider = linear_sizes[-1]//conv_features_sizes[0]
        desired_shape = torch.reshape(temp_tensor, (linear_sizes[-1]//divider, linear_sizes[-1]//divider, -1)).shape
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

class ConvolutionalEncoder(nn.Module):
    """ Class for conv encoder """
    def __init__(self, conv_features_sizes, linear_sizes, latent_size):
        super().__init__()
        
        self.conv = nn.Sequential()
        self.mlp = nn.Sequential()
        self.flat = nn.Flatten()

        self.conv.add_module(name=f"e-fconv{0}", module=_conv2d_block(1, conv_features_sizes[0], kernel_size=3, padding=1))
        self.conv.add_module(name=f"e-max{0}", module=nn.MaxPool2d(2, 2))
        for i, (in_size, out_size) in enumerate(zip(conv_features_sizes[:-1], conv_features_sizes[1:]), 1):
            self.conv.add_module(name=f"e-fconv{i}", module=_conv2d_block(in_size, out_size, kernel_size=3, padding=1))
            self.conv.add_module(name=f"e-max{i}", module=nn.MaxPool2d(2, 2))

        for i, (in_size, out_size) in enumerate(zip(linear_sizes[:-1], linear_sizes[1:])):
            self.mlp.add_module(name=f"e-linear{i}", module=nn.Linear(in_size, out_size))
            self.mlp.add_module(name=f"e-batchnorm{i}", module=nn.BatchNorm1d(out_size))
            self.mlp.add_module(name=f"e-relu{i}", module=nn.ReLU())

        self.linear_means = nn.Linear(linear_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(linear_sizes[-1], latent_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.flat(x)
        x = self.mlp(x)

        means = self.linear_means(x)
        log_vars = self.linear_means(x)

        return means, log_vars


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
        z = _reparameterize(means, log_var)
        recon_x = self.decoder(z)

        return recon_x, means, log_var

    def inference(self, z):
        return self.decoder(z)
    
class Encoder(nn.Module):

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
    
def vae_encode(model, data):
    """ Function for encoding data with vae model """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    with torch.no_grad():
        data = torch.Tensor(data)
        data = data[:, None, :]
        mu, log_var = model.encoder(data.to(device))
        output = _reparameterize(mu, log_var).cpu().numpy()
        output = np.reshape(output, [-1, model.latent_size])
        return output
    
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

def train_vae_model(model, learning_rate, weight_decay, num_epochs, patience, 
                    dataloader_train, dataloader_val, checkpoint_name='checkpoint_vae.pth'):
    """
    Function for training model with MSE and Adam Optimizer
    :param model: model to be trained
    :param learning_rate: 
    :param weight_decat:
    :param num_epochs:
    :param patience:
    :param data_loader_train:
    :param data_loader_val:
    :param checkpoint_name:
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'VAE model training performed on {device}')

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # monitor training loss per batch
    train_loss = []
    # monitor validation loss per batch
    val_loss = []
    # save avg train losses for early stopping visualization
    avg_train_loss = []
    # save avg train losses for early stopping visualization
    avg_val_loss = []
    # counter for patience in early sotpping
    patience_counter = 0
    # best validation score
    best_val_loss = -1
    # model checkpoint filename
    checkpoint_filename = checkpoint_name
    # early stopping epoch
    win_epoch = 0

    # pass model to gpu if is available
    model.to(device)

    mu = 0.0
    log_val = 0.0
    
    for epoch in range(1, num_epochs + 1):
        ###################
        # train the model #
        ###################
        print(f'-> training at epoch {epoch}', flush=True)
        model.train()
        with tqdm(total=len(dataloader_train)) as pbar:
            for input_data, labels in tqdm(dataloader_train, position=0, leave=True):
                # transfer data to device
                batch_size = input_data.shape[0]
                input_data = input_data.to(device)
                # clear the gradients of all optimized variables
                optimizer.zero_grad()
                # forward pass
                outputs, mu, log_var = model(input_data)
                # calculate the BCE loss with KL
                loss = vae_loss(outputs, input_data, mu, log_var)
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                optimizer.step()
                # update running training loss
                train_loss.append(loss.item())
                pbar.update(1)

        ###################
        # val the model   #
        ###################
        print(f'-> validating at epoch {epoch}', flush=True)
        model.eval()
        with tqdm(total=len(dataloader_val)) as pbar:
            for input_data_val, labels in tqdm(dataloader_val, position=0, leave=True):
                # transfer data to device
                batch_size_val = input_data_val.shape[0]
                input_data_val = input_data_val.to(device)
                # forward pass
                val_outputs, val_mu, val_log_var  = model(input_data_val)
                # calculate the BCE loss with KL
                vloss = vae_loss(val_outputs, input_data_val, val_mu, val_log_var)
                # update running val loss
                val_loss.append(vloss.item())
                pbar.update(1)

        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_loss)
        val_loss = np.average(val_loss)
        avg_train_loss.append(train_loss)
        avg_val_loss.append(val_loss)

        epoch_len = len(str(num_epochs))
        # print avg training statistics
        print(
            f'[{epoch:>{epoch_len}}/{num_epochs:>{epoch_len}}] train_loss: {train_loss:.5f} valid_loss: {val_loss:.5f}',
            end=' ', flush=True)

        if val_loss < best_val_loss or best_val_loss == -1:
            # new checkpoint
            print("checkpoint!")
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_filename)
            win_epoch = epoch
        elif patience_counter >= patience:
            print("early stopping.")
            print(f"=> loading checkpoint {checkpoint_filename}")
            model.load_state_dict(torch.load(checkpoint_filename))
            break
        else:
            print(".")
            patience_counter = patience_counter + 1

        # clear batch losses
        train_loss = []
        val_loss = []
        
    fig = plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, epoch + 1), avg_train_loss, 'r', label="train loss")
    plt.plot(np.arange(1, epoch + 1), avg_val_loss, 'b', label="validation loss")
    plt.axvline(win_epoch, linestyle='--', color='g', label='Early Stopping Checkpoint')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

    return model
