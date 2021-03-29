import os
import torch
import random 
import numpy as np

from math import sqrt
from torch.utils.data import Dataset
from scipy.fftpack import fft, fftfreq
from scipy import signal, fftpack
from sklearn.preprocessing import MinMaxScaler

from utils.dataset.sound import Sound

class PeriodogramDataset(Dataset, Sound):
    """ Periodogram dataset """
    def __init__(self, filenames, hives, scale_db=False, scale=False, slice_freq=None):
        Sound.__init__(self, filenames, hives)
        self.slice_freq = slice_freq
        self.scale = scale          # should scale data to be within 0 and 1
        self.scale_db = scale_db    # should scale data to log amplitude

    def get_params(self):
        """ Function for returning params """
        return {
            'slice_freq_start': self.slice_freq[0],
            'slice_freq_stop': self.slice_freq[1],
            'scaled_db': self.scale_db,
            'scaled': self.scale
        }
        
    def __getitem__(self, idx):
        """ Method for pytorch dataloader """
        (periodogram, _), labels = self.get_item(idx)
        return [periodogram[None, :]], labels

    def get_item(self, idx):
        """ Function for getting periodogram """
        # read sound samples from file
        sound_samples, sampling_rate, labels = Sound.read_sound(self, idx=idx, raw=True)

        periodogram = abs(np.fft.rfft(sound_samples, sampling_rate))[1:]
        if self.scale_db:
            periodogram = 20*np.log10(periodogram/np.iinfo(sound_samples[0]).max)
        frequencies = np.fft.rfftfreq(sampling_rate, d=(1./sampling_rate))[1:]

        if self.slice_freq:
            periodogram = periodogram[self.slice_freq[0]:self.slice_freq[1]]
            frequencies = frequencies[self.slice_freq[0]:self.slice_freq[1]]

        if self.scale:
            periodogram = MinMaxScaler().fit_transform(periodogram.reshape(-1, 1)).squeeze()

        periodogram = periodogram.astype(np.float32)
        return (periodogram, frequencies), labels
        
    def __len__(self):
        return len(self.filenames)