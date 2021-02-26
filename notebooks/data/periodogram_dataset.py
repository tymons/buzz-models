import os
import torch
import random 
import numpy as np

from math import sqrt
from torch.utils.data import Dataset
from scipy.fftpack import fft, fftfreq
from sklearn.preprocessing import MinMaxScaler
from data.sound import Sound

class PeriodogramDataset(Dataset, Sound):
    """ Periodogram dataset """
    def __init__(self, filenames, hives, slice_freq=None):
        Sound.__init__(self, filenames, hives)
        self.slice_freq = slice_freq

    def __getitem__(self, idx):
        # read sound samples from file
        sound_samples, sampling_rate, label = Sound.read_sound(self, idx)

        # calculate periodogram
        periodogram = fft(sound_samples, n=sampling_rate)
        periodogram = abs(periodogram[1:int(len(periodogram)/2)])
        if self.slice_freq:
            periodogram = periodogram[self.slice_freq[0]:self.slice_freq[1]]
        scaled_perio = MinMaxScaler().fit_transform(periodogram.reshape(-1, 1)).T
        scaled_perio = scaled_perio.astype(np.float32)
        return (scaled_perio, label)
        
    def __len__(self):
        return len(self.files)