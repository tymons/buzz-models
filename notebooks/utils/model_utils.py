import os
import time
import logging

from comet_ml import Experiment

import torch
import matplotlib.pyplot as plt
import numpy as np

from torch.utils import data as tdata
from torch import nn
from utils.data_utils import batch_normalize, batch_standarize
from utils.models.vae import vae_loss


def train_model(model, learning_params, train_loader, val_loader, comet_model_params=None, comet_tag_list=None):
    """ Function for training model """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = type(model).__name__.lower()
    logging.info(f'{model_name} model training performed on {device}')

    # prepare loss function
    loss_fun = {
        'autoencoder': nn.MSELoss(),
        'convolutionalae': nn.MSELoss(),
        'vae': vae_loss,
        'convolutionalvae': vae_loss,
        'cvae': nn.MSELoss(),
        'convolutionalcvae': nn.MSELoss(),
    }.get(model_name, lambda x: logging.error(f'loss function for model {x} not implemented!'))

    # setup comet ml experiment
    experiment = Experiment(project_name=f"{model_name.lower()}-bee-sound", auto_metric_logging=False)
    experiment.log_parameters(learning_params)
    if comet_model_params:
        experiment.log_parameters(comet_model_params)
    experiment.set_name(f"{model_name}-{time.strftime('%Y%m%d-%H%M%S')}")
    if comet_tag_list:
        experiment.add_tags(comet_tag_list)

    # fixed adam optimizer with hyperparameters
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_params['learning_rate'], weight_decay=learning_params['weight_decay'])

    # monitor training loss per batch
    train_loss = []
    # monitor validation loss per batch
    val_loss = []
    # counter for patience in early sotpping
    patience_counter = 0
    # best validation score
    best_val_loss = -1
    # model checkpoint filename
    checkpoint_filename = f'output{os.sep}models{os.sep}{experiment.get_name()}-checkpoint.pth'
    # early stopping epoch
    win_epoch = 0

    # pass model to gpu if is available
    model.to(device)

    for epoch in range(1, learning_params['epochs'] + 1):
        ###################
        # train the model #
        ###################
        model.train()
        step = 0
        for data, label in train_loader:
            # transfer data to device and normalize per batch
            if learning_params['batch_standarize']:
                data = batch_standarize(data)

            if learning_params['batch_normalize']:
                data = batch_normalize(data)

            input_data = data.to(device)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass
            output_dict = model(input_data)
            # calculate the loss
            loss = loss_fun(input_data, **output_dict)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            train_loss.append(loss.item())
            # update metric to comet
            step = step + 1
            experiment.log_metric("batch_train_loss", loss.item(), step=step)

        ###################
        # val the model   #
        ###################
        model.eval()
        step = 0
        for val_data, label in val_loader:
            # transfer data to device and normalize per batch
            if learning_params['batch_standarize']:
                val_data = batch_standarize(val_data)

            if learning_params['batch_normalize']:
                val_data = batch_normalize(val_data)
            
            # transfer data to device
            input_data_val = val_data.to(device)
            # forward pass
            val_output_dict = model(input_data_val)
            # calculate the loss
            vloss = loss_fun(input_data_val, **val_output_dict)
            # update running val loss
            val_loss.append(vloss.item())
            # update metric to comet
            step = step + 1
            experiment.log_metric("batch_val_loss", vloss.item(), step=step)

        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_loss)
        val_loss = np.average(val_loss)

        # print avg training statistics
        logging.info(f'Epoch [{epoch}/{learning_params["epochs"]}], LOSS: {train_loss:.6f}, VAL_LOSS: {val_loss:.6f}')
        experiment.log_metric("train_loss", train_loss, step=epoch)
        experiment.log_metric("val_loss", val_loss, step=epoch)

        if val_loss < best_val_loss or best_val_loss == -1:
            # new checkpoint
            logging.info("checkpoint!")
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_filename)
            win_epoch = epoch
        elif patience_counter >= learning_params['patience']:
            logging.info("early stopping.")
            logging.info(f"=> loading checkpoint {checkpoint_filename}")
            model.load_state_dict(torch.load(checkpoint_filename))
            break
        else:
            print(".")
            patience_counter = patience_counter + 1

        # clear batch losses
        train_loss = []
        val_loss = []