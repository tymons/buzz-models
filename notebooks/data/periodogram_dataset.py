import os
import torch
import random 
import numpy as np

from math import sqrt
from torch.utils.data import Dataset
from scipy.fftpack import fft, fftfreq
from data.sound import Sound
from scipy import signal, fftpack

class PeriodogramDataset(Dataset, Sound):
    """ Periodogram dataset """
    def __init__(self, filenames, hives, scale_db=False, slice_freq=None):
        Sound.__init__(self, filenames=filenames, labels=hives)
        print(f'params: scale_db({scale_db}), slice_freq({slice_freq})')
        self.slice_freq = slice_freq
        self.scale_db = scale_db

    def __getitem__(self, idx):
        # read sound samples from file
        sound_samples, sampling_rate, label = Sound.read_sound(self, idx=idx, raw=True)

        periodogram = abs(np.fft.rfft(sound_samples, sampling_rate))[1:]
        if self.scale_db:
            periodogram = 20*np.log10(periodogram/np.iinfo(sound_samples[0]).max)
        frequencies = np.fft.rfftfreq(sampling_rate, d=(1./sampling_rate))[1:]

        if self.slice_freq:
            periodogram = periodogram[self.slice_freq[0]:self.slice_freq[1]]
            frequencies = frequencies[self.slice_freq[0]:self.slice_freq[1]]

        periodogram = periodogram.astype(np.float32)
        return periodogram, frequencies
        
    def __len__(self):
        return len(self.files)