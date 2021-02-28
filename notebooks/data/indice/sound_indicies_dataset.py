import os
import data.indice.compute_indice as ci

from data.indice.acoustic_index import AudioFile
from enum import Enum
from torch.utils.data import Dataset

import matplotlib.pyplot as plt

class SoundIndiciesDataset(Dataset):
    def __init__(self, filenames, hives, indicator_type, n_fft, hop_len, j_samples):
        self.filenames = filenames
        self.labels = hives
        self.indicator_type = indicator_type
        self.nfft = n_fft
        self.hop_len = hop_len
        self.j_samples = j_samples

    def __getitem__(self, idx):

        def get_ACI(sound_file):
            spectro, frequencies = ci.compute_spectrogram(sound_file, self.nfft, self.hop_len)
            print(f'shape of spectrum: {spectro.shape}')
            return ci.compute_ACI(spectro, self.j_samples)

        def get_ADI(sound_file):
            pass

        def get_AEI(sound_file):
            pass

        def get_BI(sound_file):
            pass

        # read sound samples from file
        filename = self.filenames[idx]
        sound_file = AudioFile(filename) 
        hive_name = filename.split(os.sep)[-2].split("_")[0]
        try:
            label = next(index for index, name in enumerate(self.labels) if name == hive_name)
        except StopIteration as e:
            label = -1

        feature, temporal = {
            self.SoundIndicator.ACI:    get_ACI,  
            self.SoundIndicator.ADI:    get_ADI, 
            self.SoundIndicator.AEI:    get_AEI,
            self.SoundIndicator.BI:    get_BI,
        }.get(self.indicator_type)(sound_file)

        print(f'feature: {feature} / temporal shape: {temporal}')
        return (feature, label)
        
    class SoundIndicator(Enum):
        ACI = 1     # acoustic complexity index
        ADI = 2     # acoustic diversity index
        AEI = 3     # acoustic evenness index
        BI = 4      # bioacustic index