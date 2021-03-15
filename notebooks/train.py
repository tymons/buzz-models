#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import argparse
import logging
import json
import ast

from colorama import init, deinit, Fore, Back, Style

from utils.data_utils import create_valid_sounds_datalist, get_valid_sounds_datalist
from utils.features import SoundFeatureDataset

import matplotlib.pyplot as plt

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

    return sound_filenames, labels

def main():
    if os.name == 'nt':
        init()      # colorama init stdout and stderr as win32 system calls

    parser = argparse.ArgumentParser(description='Process some integers.')
    # positional arguments
    parser.add_argument('model_type', metavar='model_type', type=str, help='Model Type')
    parser.add_argument('feature', metavar='feature', type=str, help='Input feature')
    parser.add_argument('root_folder', metavar='root_folder', type=str, help='Root folder for data')
    # optional arguments
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
    logging.info(f'starting smartula experiment for {ModelType(args.model_type.lower())} model and {InputType(args.input_type.lower())}.')

    # read sound filenames from 'valid-files.txt' files
    sound_filenames, labels = get_soundfilenames_and_labels(args.root_folder, 'valid-files.txt', args.check_data)
    # get train and val loaders
    train_loader, val_loader = SoundFeatureDataset.get_dataloaders(args.feature, sound_filenames, labels, config['model'].get('batch_size', 32), config['features'])

    if os.name == 'nt':
        deinit()      # colorama resotore
if __name__ == "__main__":
    main()