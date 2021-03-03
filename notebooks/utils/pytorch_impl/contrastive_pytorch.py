import torch 
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

from torch import nn
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils.pytorch_impl.vae import vae_loss
   
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
            for input_data, labels in tqdm(dataloader_train, position=0, leave=True):
                if scale:
                    # scale data accross batch
                    input_data[:, 0, :] = torch.Tensor(MinMaxScaler().fit_transform(StandardScaler().fit_transform(input_data[:, 0, :])))

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
            for input_data_val, labels in tqdm(dataloader_val, position=0, leave=True):
                if scale:
                    # scale data accross batch
                    input_data_val[:, 0, :] = torch.Tensor(MinMaxScaler().fit_transform(StandardScaler().fit_transform(input_data_val[:, 0, :])))

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
