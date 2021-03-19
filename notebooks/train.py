#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import argparse
import logging
import json
import ast
import comet_ml
import torch

from colorama import init, deinit, Back

from utils.data_utils import create_valid_sounds_datalist, get_valid_sounds_datalist
from utils.feature_factory import SoundFeatureFactory
from utils.model_factory import HiveModelFactory
from utils.model_utils import train_model
from utils.data_utils import filter_strlist

def truncate_lists_to_smaller_size(arg1, arg2):
    if len(arg1) > len(arg2):
        arg1 = arg1[:len(arg2)]
    else:
        arg2 = arg2[:len(arg1)]

    return arg1, arg2


def get_soundfilenames_and_labels(root_folder: str, valid_sounds_filename: str, data_check_reinit: bool):
    """ Function for getting soundlist from root folder """
    if data_check_reinit:
        checked_folders = create_valid_sounds_datalist(root_folder, valid_sounds_filename, "smrpiclient")
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

    assert (len(sound_filenames) > 0), Back.RED + "we cannot read any data from specified folder!"

    return sound_filenames, list(labels)

def main():
    if os.name == 'nt':
        init()      # colorama init stdout and stderr as win32 system calls

    parser = argparse.ArgumentParser(description='Process some integers.')
    # positional arguments
    parser.add_argument('model_type', metavar='model_type', type=str, help='Model Type [vae, cvae, contrastive_vae, contrastive_cvae, ae, cae]')
    parser.add_argument('feature', metavar='feature', type=str, help='Input feature')
    parser.add_argument('root_folder', metavar='root_folder', type=str, help='Root folder for data')
    # optional arguments
    parser.add_argument("--background", type=str, nargs='+', help="folder prefixes for background data in contrastive learning")
    parser.add_argument("--target", type=str, nargs='+', help="folder prefixes for target data in contrastive learning")
    parser.add_argument("--discriminator", type=bool, default=False, help="should use discirminator in contrastive learning")
    parser.add_argument("--check_data", type=bool, default=False, help="should check sound data")
    parser.add_argument("--log_file", type=str, default='debug.log', help="name of debug file")
    parser.add_argument("--config_file", type=str)
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
            logging.FileHandler(args.log_file),
            logging.StreamHandler()
        ]
    )
    logging.info(f'starting smartula experiment for {args.model_type} model and {args.feature}.')

    # read sound filenames from 'valid-files.txt' files
    sound_filenames, labels = get_soundfilenames_and_labels(args.root_folder, 'valid-files.txt', args.check_data)
    target_filenames, target_labels = (filter_strlist(sound_filenames, *args.target), args.target) if args.target else (sound_filenames, labels)
    background_filenames, background_labels = (filter_strlist(sound_filenames, *args.background), args.background) if args.background else (None, None)
    if background_filenames is not None:
        # so we have contrastive learning and target/background has to have the same size
        target_filenames, background_filenames = truncate_lists_to_smaller_size(target_filenames, background_filenames)
        logging.info(f'got {len(target_filenames)} files as target data and {len(background_filenames)} as background for contrastive learning')
    
    # get loaders
    train_loader, val_loader = SoundFeatureFactory.build_dataloaders(args.feature, target_filenames, target_labels, 
                                                        config['features'], config['learning'].get('batch_size', 32),
                                                        background_filenames=background_filenames, background_labels=background_labels)
    # get model
    model, model_params = HiveModelFactory.build_model(args.model_type, config['model_architecture'][args.model_type], train_loader.dataset[0][0][0].shape)
    discirminator, disc_params = HiveModelFactory.build_model('discriminator', config['model_architecture']['discriminator'],
                                                    config['model_architecture'][args.model_type]['latent_size'] * 2) if args.discriminator else (None, {})
    discirminator_alpha = config['learning'].get('discriminator_alpha', 0.1) if args.discriminator else None
    
    # train model
    model_params.update(disc_params)
    labels.append(args.feature)
    model = train_model(model, config['learning'], train_loader, val_loader, discriminator=discirminator,
                         comet_params=model_params, comet_tags=labels)

    if os.name == 'nt':
        deinit()      # colorama restore

if __name__ == "__main__":
    main()