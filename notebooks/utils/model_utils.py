import os
import time
import logging

from comet_ml import Experiment

import torch
import numpy as np
from torch.utils import data as tdata
from torch import nn

from utils.data_utils import batch_normalize, batch_standarize
from utils.models.vae import vae_loss
from utils.models.discriminator import discriminator_loss
from utils.models.ae import ae_loss_fun
from utils.models.cvae import cvae_loss

def permutate_latent(latents_batch, inplace=False):
    """ Function for element permutation along axis
    
    Parameters:
        latent_batch (torch.tensor): input matrix to be permutated
        inplace (bool): modify original tensor or not
    Returns
    """
    latents_batch = latents_batch.squeeze()
    
    data = latents_batch.detach().clone() if inplace == False else latents_batch

    for column_idx in range(latents_batch.shape[-1]):
        rand_indicies = torch.randperm(latents_batch[:, column_idx].shape[0])
        latents_batch[:, column_idx] = latents_batch[:, column_idx][rand_indicies]

    return data


def setup_comet_ml_experiment(project_name, experiment_name, parameters, tags):
    """ Function for setting up comet ml experiment """
    experiment = Experiment(project_name=project_name, auto_metric_logging=False)
    experiment.set_name(experiment_name)
    experiment.log_parameters(parameters)
    experiment.add_tags(tags)
    return experiment


def train_model(model, learning_params, train_loader, val_loader, discriminator=None,
                    comet_params={}, comet_tags=[]):
    """ Main function for training model 
    
    Parameters:
        model (torch.nn.Module): model to be trained
        learning_prams (dict): learning dictionary from json config
        train_loader (DataLoader): data loader for train data
        val_loader (DataLoader): data loader for validation data
        discriminator (nn.Module): discriminator for contrastive learning
        comet_params (dict): parameters which will be uploaded to comet ml
        comet_tags (list): list of tags which should be uploaded to comet ml experiment

    Returns
        model (torch.nn.Module): trained model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = type(model).__name__.lower()
    logging.info(f'{model_name} model training starting on {device} device')

    # prepare loss function
    loss_fun = {
        'autoencoder': ae_loss_fun,
        'convolutionalae': ae_loss_fun,
        'vae': vae_loss,
        'convolutionalvae': vae_loss,
        'cvae': cvae_loss,
        'convolutionalcvae': cvae_loss,
    }.get(model_name, lambda x: logging.error(f'loss function for model {x} not implemented!'))

    # setup comet ml experiment
    comet_tags = comet_tags.append('discriminator') if discriminator else comet_tags
    experiment = setup_comet_ml_experiment(f"{model_name.lower()}-bee-sound", f"{model_name}-{time.strftime('%Y%m%d-%H%M%S')}",
                                            parameters=dict(learning_params, **comet_params), tags=comet_tags)

    # fixed adam optimizer with hyperparameters
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_params['learning_rate'], weight_decay=learning_params['weight_decay'])
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=learning_params['discriminator']['learning_rate'],
                                                 weight_decay=learning_params['discriminator']['weight_decay']) if discriminator else None
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
    if discriminator is not None:
        discriminator.to(device)
        discriminator.train()

    for epoch in range(1, learning_params['epochs'] + 1):
        ###################
        # train the model #
        ###################
        model.train()
        step = 1
        for data, label in train_loader:
            # batch calculation
            if learning_params['batch_standarize']:
                data = batch_standarize(data)
            if learning_params['batch_normalize']:
                data = batch_normalize(data)
            input_data = data.to(device)

            optimizer.zero_grad()
            output_dict = model(input_data)
            loss = loss_fun(input_data, output_dict)
            loss.backward()
            optimizer.step()
            # log comet ml metric and watch avg losses
            experiment.log_metric("batch_train_loss", loss.item(), step=step)
            train_loss.append(loss.item())

            if discriminator:
                # if we have contrastive learning which supports discriminator
                q = torch.cat((output_dict["latent_qs_target"].detach(), output_dict["latent_qz_target"].detach()), axis=-1).to(device)
                q_bar = permutate_latent(q)
                q_bar = q_bar.to(device)
                optimizer_discriminator.zero_grad()
                q_score, q_bar_score = discriminator(q, q_bar)
                dloss = discriminator_loss(q_score, q_bar_score)
                dloss.backward()
                optimizer_discriminator.step()
                # log comet ml metric
                experiment.log_metric("discriminator_train_loss", dloss.item(), step=step)

            step = step + 1

        ###################
        # val the model   #
        ###################
        model.eval()
        step = 1
        for val_data, label in val_loader:
            if learning_params['batch_standarize']:
                val_data = batch_standarize(val_data)
            if learning_params['batch_normalize']:
                val_data = batch_normalize(val_data)
            input_data_val = val_data.to(device)
            
            val_output_dict = model(input_data_val)
            vloss = loss_fun(input_data_val, **val_output_dict)
            val_loss.append(vloss.item())
            # log comet ml metric
            experiment.log_metric("batch_val_loss", vloss.item(), step=step)
            step = step + 1

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
            patience_counter = patience_counter + 1

        # clear batch losses
        train_loss = []
        val_loss = []