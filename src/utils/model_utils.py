import os
import time
import logging
import math
import random

from comet_ml import Experiment

import torch
import numpy as np
from torch.utils import data as tdata
from torch import nn

from .data_utils import batch_normalize, batch_standarize, batch_addnoise
from .models.vae import vae_loss
from .models.conv_ae import conv_mlp_layer_shape
from .models.discriminator import discriminator_loss
from .models.ae import ae_loss_fun
from .models.cvae import cvae_loss


def generate_discriminator_model_config(range_config):
    """ Function for generating discriminator model config """
    config = {
        'layers': []
    }
    fc_layers = []

    # setup random generator
    random.seed()

    number_of_fc_layers = random.randint(range_config['layers_number_range'][0], range_config['layers_number_range'][1])
    
    fc_layer = range_config['layer_size_range'][1]
    for layer_no in range(number_of_fc_layers):
        fc_layer = random.randint(range_config['layer_size_range'][0], fc_layer)
        fc_layers.append(fc_layer)

    config['layers'] = fc_layers

    return config

def generate_train_infos(range_config):
    """ Function for generating learning config. Note that here we generate partial
        train config. Result of this function should be merged with real config e.g. config['learning'] """
    config = {
        'batch_size': 0,
        'learning_rate': 0,
        'discriminator': {
            'alpha': 0,
            'learning_rate': 0
        }
    }

    # setup random generator
    random.seed()

    config['batch_size'] = random.randint(range_config['batch_size_range'][0], range_config['batch_size_range'][1])
    config['learning_rate'] = 1/(10**random.randint(range_config['learning_rate_order_range'][0], range_config['learning_rate_order_range'][1]))
    config['discriminator']['alpha'] = random.uniform(range_config['discriminator']['alpha_range'][0], range_config['discriminator']['alpha_range'][1])
    config['discriminator']['alpha'] = 1/(10**random.randint(range_config['discriminator']['learning_rate_order_range'][0], \
                                                            range_config['discriminator']['learning_rate_order_range'][1]))

    return config


def generate_fc_model_config(range_config):
    """ Fucntion for generating fully connected model config """
    config = {
        'encoder_layer_sizes': [],
        'decoder_layer_sizes': [],
        'latent_size': 0
    }

    # setup random generator
    random.seed()

    number_of_fc_layers = random.randint(range_config['layers_number_range'][0], range_config['layers_number_range'][1])

    fc_layers = []
    fc_max_size = range_config['layer_size_range'][1]
    for layer_no in range(number_of_fc_layers):
        fc_max_size = random.randint(range_config['layer_size_range'][0], fc_max_size)
        fc_layers.append(fc_max_size)
    fc_layers = list(dict.fromkeys(fc_layers)) # remove duplicates

    latent_size = random.randint(range_config['latent_size_range'][0], range_config['latent_size_range'][1])

    config['encoder_layer_sizes'] = fc_layers
    config['decoder_layer_sizes'] = fc_layers[::-1]
    config['latent_size'] = latent_size

    return config


def generate_conv_model_config(range_config, input_shape):
    """ Function for generating convolutional model config """
    config = {
        'encoder_feature_maps': [],
        'encoder_mlp_layer_sizes': [],
        'decoder_feature_maps': [],
        'decoder_mlp_layer_sizes': [],
        'latent_size': 0
    }
    
    # setup random generator
    random.seed()

    # convolutional layers generation
    conv_feature_map = []

    while True:
        number_of_conv_layers = random.randint(range_config['conv_layers_number_range'][0], range_config['conv_layers_number_range'][1])
        conv_max_size = range_config['conv_features_range'][1]
        for layer in range(number_of_conv_layers):
            conv_max_size = random.randint(range_config['conv_features_range'][0], conv_max_size)
            conv_feature_map.append(conv_max_size)
        conv_feature_map = list(dict.fromkeys(conv_feature_map)) # remove duplicates

        # check if we don't have too many layers for given input shape
        if all(conv_mlp_layer_shape(input_shape, conv_feature_map, kernel=3, stride=1, padding=1, max_pool=(2,2))):
            break
        else:
            logging.warning('too many layers for convolutional nn - try to modify "conv_layers_number_range" from random'
                            ' search config')

    # fully connected layers generation
    mlp_layer_sizes = []
    number_of_mlp_layers = random.randint(range_config['mlp_layers_number_range'][0], range_config['mlp_layers_number_range'][1])    
    layer_size = range_config['mlp_layer_size_range'][1]
    for layer in range(number_of_mlp_layers):
        layer_size = random.randint(range_config['mlp_layer_size_range'][0], layer_size)
        mlp_layer_sizes.append(layer_size)
    mlp_layer_sizes = list(dict.fromkeys(mlp_layer_sizes)) # remove duplicates

    # latent generation
    latent_size = random.randint(range_config['latent_size_range'][0], range_config['latent_size_range'][1])
    
    config['encoder_feature_maps'] = conv_feature_map
    config['decoder_feature_maps'] = conv_feature_map[::-1]
    config['encoder_mlp_layer_sizes'] = mlp_layer_sizes
    config['decoder_mlp_layer_sizes'] = mlp_layer_sizes[::-1]
    config['latent_size'] = latent_size

    return config


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


def setup_comet_ml_experiment(api_key, project_name, experiment_name, parameters, tags):
    """ Function for setting up comet ml experiment """
    experiment = Experiment(api_key=api_key, project_name=project_name, auto_metric_logging=False)
    experiment.set_name(experiment_name)
    experiment.log_parameters(parameters)
    experiment.add_tags(tags)
    return experiment

def _model_save(model, optimizer, loss, epoch, checkpoint_full_path, discriminator=None, discriminator_optimizer=None):
    """ Function for saving model on disc """
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'discriminator_state_dict': discriminator.state_dict() if discriminator else None,
                'discriminator_optimizer_state_dict': discriminator_optimizer.state_dict() if discriminator_optimizer else None
                }, checkpoint_full_path)

def model_load(checkpoint_filepath, model, optimizer=None, discriminator=None, discriminator_optimizer=None):
    """ Function for loading checkpoint 
    
    Parameters:
        checkpoint_filepath (str): filepath to checkpoint
        model (nn.Module): object model where weights will be loaded
        optimizer (Optimizer): optimizer
        discriminator (nn.Module): discriminator 
        discriminator_optimizer (Optimizer): optimizer for discriminator
    Returns:
        (int, float) (tuple): last epoch and last loss
    """
    checkpoint = torch.load(checkpoint_filepath)
    if any([key.startswith('module') for key in checkpoint['model_state_dict'].keys()]):
        # conver model to dataparallel if we trained our model on many gpus
        model = nn.DataParallel(model)

    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    if discriminator and discriminator_optimizer:
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])

    return epoch, loss

def _batch_transform(batch, standarize, normalize):
    """ Wrapper for performing batch specific transformations 
        Note that adding noise performs clipping as well, data should be within range [0,1]
    Parameters:
        batch
        standarize (bool):
        normalize (bool)
        add_nosize (bool)
        noise_factor (float):

    Returns:
        batch: transformated batc 
    """
    if standarize:
        batch = batch_standarize(batch)
    if normalize:
        batch = batch_normalize(batch)
    return batch

def train_model(model, learning_params, train_loader, val_loader, discriminator=None, denoising=False,
                    comet_params={}, comet_tags=[], model_output_folder="output", comet_api_key=None):
    """ Main function for training model 
    
    Parameters:
        model (torch.nn.Module): model to be trained
        learning_prams (dict): learning dictionary from json config
        train_loader (DataLoader): data loader for train data
        val_loader (DataLoader): data loader for validation data
        discriminator (nn.Module): discriminator for contrastive learning
        denoising (bool): flag if we should use autoencoder in denosing manner
        comet_params (dict): parameters which will be uploaded to comet ml
        comet_tags (list): list of tags which should be uploaded to comet ml experiment
        model_output_folder (str): folder where output models will be saved
        comet_api_key (str): api key for comet ml if is None .comet.config should be available in src directory

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
    learning_params_log = {f"LEARNING_{key}": val for key, val in learning_params.items()}
    experiment = setup_comet_ml_experiment(comet_api_key, f"{model_name.lower()}-bee-sound", f"{model_name}-{time.strftime('%Y%m%d-%H%M%S')}",
                                            parameters={**learning_params_log, **comet_params}, tags=comet_tags)

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
    checkpoint_file = f'{experiment.get_name()}-checkpoint.pth'
    checkpoint_full_path = os.path.join(model_output_folder, checkpoint_file)

    # pass model to gpu if is available
    if torch.cuda.device_count() > 1:
        logging.info(f"we will be using {torch.cuda.device_count()} GPUs.")
        model = nn.DataParallel(model)

    model.to(device)
    if discriminator is not None:
        discriminator.to(device)

    for epoch in range(1, learning_params['epochs'] + 1):
        ###################
        # train the model #
        ###################
        step = 1
        model.train()
        if discriminator:
            discriminator.train()

        for concatenated_batch, label in train_loader:
            concatenated_input_batch = [None] * len(concatenated_batch)
            for position, batch in enumerate(concatenated_batch):
                # as every dataset should return list (one or two elements - depends on contrastive learning or not)
                # we should every element normalize/standarzie or add noise for denosing autoencoder and pass to gpu
                transformed_batch = _batch_transform(batch, learning_params['batch_standarize'], learning_params['batch_normalize'])
                concatenated_batch[position] = transformed_batch.to(device)
                if denoising:
                    concatenated_input_batch[position] = batch_addnoise(transformed_batch, learning_params['denoising_factor']).to(device)
                else:
                    concatenated_input_batch[position] = concatenated_batch[position]

            optimizer.zero_grad()
            output_dict = model(*concatenated_input_batch)      
            if discriminator is None:
                loss = loss_fun(*concatenated_batch, output_dict)
            else:
                loss = loss_fun(*concatenated_batch, output_dict, discriminator=discriminator, discriminator_alpha=learning_params['discriminator']['alpha'])
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
        for concatenated_val_batch, label in val_loader:
            concatenated_input_val_batch = [None] * len(concatenated_val_batch)
            for position, batch_val in enumerate(concatenated_val_batch):
                # as every dataset should return list (one or two elements - depends on contrastive learning or not)
                # we should every element normalize/standarzie and pass to gpu
                transformed_val_batch = _batch_transform(batch_val, learning_params['batch_standarize'], learning_params['batch_normalize'])
                concatenated_val_batch[position] = transformed_val_batch.to(device)
                if denoising:
                    concatenated_input_val_batch[position] = batch_addnoise(transformed_val_batch, learning_params['denoising_factor']).to(device)
                else:
                    concatenated_input_val_batch[position] = concatenated_val_batch[position]
            
            val_output_dict = model(*concatenated_input_val_batch)
            if discriminator is None:
                vloss = loss_fun(*concatenated_val_batch, val_output_dict)
            else:
                vloss = loss_fun(*concatenated_val_batch, val_output_dict, discriminator=discriminator, discriminator_alpha=learning_params['discriminator']['alpha'])
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
            _model_save(model, optimizer, loss, epoch, checkpoint_full_path, discriminator=discriminator, \
                                                                                discriminator_optimizer=optimizer_discriminator)
        elif patience_counter >= learning_params['patience'] or val_loss is math.isnan(val_loss):
            logging.info("early stopping!")
            break
        else:
            patience_counter = patience_counter + 1

        # clear batch losses
        train_loss = []
        val_loss = []
    
    epoch, _ = model_load(checkpoint_full_path, model, optimizer, discriminator=discriminator, \
                                                            discriminator_optimizer=optimizer_discriminator)
    logging.info(f"=> loaded model based on {checkpoint_full_path} checkpoint file, saved on {epoch} epoch.")
    return model

