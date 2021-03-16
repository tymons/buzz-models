import torch
import torch.nn.functional as F

from tqdm import tqdm
from torch import nn
from utils.data_utils import batch_normalize, batch_standarize
from utils.models.ae import Encoder, Decoder


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
        x = self.encoder(x)
        means = self.linear_means(x)
        log_var =  self.linear_log_var(x)
        z = reparameterize(means, log_var)
        recon_x = self.decoder(z)

        return recon_x, means, log_var
    

def train_vae_model(model, learning_rate, weight_decay, num_epochs, patience, 
                    dataloader_train, dataloader_val, scale=False, checkpoint_name='checkpoint_vae.pth'):
    """
    Function for training model with MSE and Adam Optimizer
    :param model: model to be trained
    :param learning_rate: 
    :param weight_decat:
    :param num_epochs:
    :param patience:
    :param data_loader_train:
    :param data_loader_val:
    :param scale: if we should scale accross batch
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
            for input_data in tqdm(dataloader_train, position=0, leave=True):
                if scale:
                    # scale data accross batch
                    input_data = batch_normalize(batch_standarize(input_data))

                # transfer data to device
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
            for input_data_val in tqdm(dataloader_val, position=0, leave=True):
                if scale:
                    # scale data accross batch
                    input_data_val = batch_normalize(batch_standarize(input_data_val))

                # transfer data to device
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