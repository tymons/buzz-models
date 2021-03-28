#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import argparse
import logging
import json
import comet_ml
import torch
import utils.model_utils as m
import traceback

from datetime import datetime
from utils.data_utils import create_valid_sounds_datalist, get_valid_sounds_datalist
from utils.feature_factory import SoundFeatureFactory
from utils.model_factory import HiveModelFactory

from utils.data_utils import filter_strlist, truncate_lists_to_smaller_size, read_comet_api_key


def build_and_train_model(model_type, model_config, train_config, train_loader, val_loader,  model_output_folder,
                             use_discriminator=False, discirminator_config=None, comet_tags=[], comet_api_key=None):
    """ function for building and training model """
    data_shape = train_loader.dataset[0][0][0].squeeze().shape
    logging.info(f'building model with data input shape of {data_shape}')

    model, model_params = HiveModelFactory.build_model(model_type, model_config, data_shape)

    if model:
        discriminator, disc_params, discirminator_alpha  = (None, {}, None)
        if use_discriminator is True and discirminator_config:
            # if we have discirminator arg build that model
            discriminator, disc_params = HiveModelFactory.build_model('discriminator', discirminator_config, tuple((model_config['latent_size'] * 2, )))
            discirminator_alpha = train_config['discriminator'].get('alpha', 0.1)
            comet_tags.append('discriminator')

        # train model
        log_dict = {**model_params, **disc_params}
        try:
            model = m.train_model(model, train_config, train_loader, val_loader, discriminator=discriminator, \
                                    comet_params=log_dict, comet_tags=comet_tags, model_output_folder=model_output_folder, comet_api_key=comet_api_key)
        except Exception:
            logging.error('model train fail!')
            logging.error(traceback.print_exc())
    else:
        logging.error('cannot build ml model.')

    return model


def get_soundfilenames_and_labels(root_folder: str, valid_sounds_filename: str, data_check_reinit: bool):
    """ Function for getting soundlist from root folder """
    if data_check_reinit:
        prefix = "smrpiclient" # TODO: to be parametrized
        logging.info(f'checking all sound files from folders with "{prefix}" prefix, it may take a while...')
        checked_folders = create_valid_sounds_datalist(root_folder, valid_sounds_filename, prefix)
    else: 
        checked_folders = [os.path.join(root_folder, "smrpiclient0_10082020-19012021"),       
                os.path.join(root_folder, "smrpiclient3_10082020-19012021"), 
                os.path.join(root_folder, "smrpiclient5_10082020-19012021"),    
                os.path.join(root_folder, "smrpiclient6_10082020-19012021"),
                os.path.join(root_folder, "smrpiclient7_10082020-19012021")]
    
    # get labels as first part of folder name
    labels = {foldername.replace(root_folder, '').split('_')[0].strip(".\\/") for foldername in checked_folders}
    # get sound filenames from specified folders
    sound_filenames = get_valid_sounds_datalist(checked_folders, valid_sounds_filename)

    assert (len(sound_filenames) > 0), "we cannot read any data from specified folder!"

    return sound_filenames, list(labels)

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    # positional arguments
    parser.add_argument('model_type', metavar='model_type', type=str, help='Model Type [vae, cvae, contrastive_vae, contrastive_cvae, ae, cae]')
    parser.add_argument('feature', metavar='feature', type=str, help='Input feature')
    parser.add_argument('root_folder', metavar='root_folder', type=str, help='Root folder for data')
    # optional arguments
    parser.add_argument("--background", type=str, nargs='+', help="folder prefixes for background data in contrastive learning")
    parser.add_argument("--target", type=str, nargs='+', help="folder prefixes for target data in contrastive learning")
    parser.add_argument('--check-data', dest='check_data', action='store_true')
    parser.add_argument("--log_folder", type=str, default='.', help="name of debug file")
    parser.add_argument("--config_file", default='config.json', type=str)
    parser.add_argument('--random_search', type=int, help='number of tries to find best architecture')
    parser.add_argument('--discriminator', dest='discriminator', action='store_true')
    parser.add_argument('--model_output', type=str, default="output/models", help="folder for model output")
    parser.add_argument("--comet_config", default='config.json', type=str)
    parser.set_defaults(discriminator=False)
    parser.set_defaults(check_data=False)

    args = parser.parse_args()

    # read config file
    f = open(args.config_file)
    config = json.load(f)
    f.close()

    # setup logger
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.log_folder, f"{args.model_type}-{args.feature}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{logging._levelToName[logging.DEBUG]}.log")),
            logging.StreamHandler()
        ]
    )
    logging.info(f'starting smartula experiment for {args.model_type} model and {args.feature}.')

    # read sound filenames from 'valid-files.txt' files
    sound_filenames, labels = get_soundfilenames_and_labels(args.root_folder, 'valid-files.txt', args.check_data)
    target_filenames, target_labels = (filter_strlist(sound_filenames, *args.target), args.target) if args.target else (sound_filenames, labels)
    background_filenames, background_labels = (filter_strlist(sound_filenames, *args.background), args.background) if args.background else ([], [])
    if background_filenames:
        # so we have contrastive learning and target/background has to have the same size
        target_filenames, background_filenames = truncate_lists_to_smaller_size(target_filenames, background_filenames)
        logging.info(f'got {len(target_filenames)} files as target data and {len(background_filenames)} as background for contrastive learning')
    logging.info(f'dataset total length: {len(target_filenames) + len(background_filenames)}')

    log_labels = target_labels + background_labels + [args.feature]

    # get loaders
    train_loader, val_loader = SoundFeatureFactory.build_dataloaders(args.feature, target_filenames, target_labels, 
                                                        config['features'], config['learning'].get('batch_size', 32),
                                                        background_filenames=background_filenames, background_labels=background_labels)

    # read comet ml api key from specified file
    comet_api_key = read_comet_api_key(args.comet_config) if args.comet_config else None

    if args.random_search:
        logging.info(f'random search architecture configuration for model {args.model_type} is active.')
        for sample_no in range(args.random_search):
            # generate model config 
            if args.model_type.startswith('conv'):
                model_config = m.generate_conv_model_config(config['random_search']['model']['conv'], train_loader.dataset[0][0][0].squeeze().shape)
            elif args.model_type != 'discriminator':
                model_config = m.generate_fc_model_config(config['random_search']['model']['fc'])
            else:
                raise ValueError(f'model {args.model_type} not supported for random search!')
            # generate random train config and merge with existing 
            train_config = {**m.generate_train_infos(config['random_search']['learning']), **config['learning']}
            # generate random discriminator config if needed
            discriminator_config = m.generate_discriminator_model_config(config['random_search']['model']['discriminator']) if args.discriminator else None

            build_and_train_model(args.model_type, model_config, train_config, train_loader, val_loader, args.model_output,
                                    use_discriminator=args.discriminator, discirminator_config=discriminator_config, comet_tags=log_labels, comet_api_key=comet_api_key)

    else:
        logging.info(f'single shot {args.model_type} configuration is active.')
        model_config = config['model_architecture'][args.model_type]
        train_config = config['learning']
        discriminator_config = config['model_architecture']['discriminator'] if args.discriminator else None

        build_and_train_model(args.model_type, model_config, train_config, train_loader, val_loader, args.model_output,
                            use_discriminator=args.discriminator, discirminator_config=discriminator_config, comet_tags=log_labels, comet_api_key=comet_api_key)

if __name__ == "__main__":
    main()