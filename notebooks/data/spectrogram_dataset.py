import os
import torch
import random 
import librosa
import numpy as np

from torch.utils.data import Dataset
from scipy.io import wavfile
from sklearn.preprocessing import MinMaxScaler

from data.sound import Sound
class SpectrogramDataset(Dataset, Sound):
    """ Spectrogram dataset """
    def __init__(self, filenames, hives, nfft, hop_len, fmax=None):
        """ Constructor for Sepctrogram Dataset

        Parameters:
            filenames (list): list of strings with filenames
            hives (list): list of strings with hive names as it will server as lables
            nfft (int): how many samples for nfft
            hop_len (int): overlapping, samples for hop to next fft
            fmax (int): constraint on maximum frequency
        """
        Sound.__init__(self, filenames, hives)
        self.nfft = nfft
        self.hop_len = hop_len
        self.fmax = fmax

    def __readitem(self, idx):
        """ Function for getting item from Spectrogram dataset

        Parameters:
            idx (int): element idx

        Returns:
            ((spectrogram, frequencies, times), label) (tuple)
        """
        # read sound samples from file
        sound_samples, sampling_rate, label = Sound.read_sound(self, idx)

        # calculate spectrogram
        spectrogram = librosa.core.stft(sound_samples, n_fft=self.nfft, hop_length=self.hop_len)
        spectrogram_magnitude = np.abs(spectrogram)
        # spectrogram_phase = np.angle(spectrogram)
        spectrogram_db = librosa.amplitude_to_db(spectrogram_magnitude, ref=np.max)
        frequencies = librosa.fft_frequencies(sr=sampling_rate, n_fft=self.nfft)
        times = (np.arange(0, spectrogram_magnitude.shape[1])*self.hop_len)/sampling_rate
        if self.fmax:
            freq_slice = np.where((frequencies < self.fmax))
            frequencies = frequencies[freq_slice]
            spectrogram_db = spectrogram_db[freq_slice, :]

        initial_shape = spectrogram_db.shape
        scaled_spectrogram_db = MinMaxScaler().fit_transform(spectrogram_db.reshape(-1, 1)).reshape(initial_shape)
        scaled_spectrogram_db = scaled_spectrogram_db.astype(np.float32)
        return ((scaled_spectrogram_db, frequencies, times), label)

    def __getitem__(self, idx):
        """ Wrapper for getting item from Spectrogram dataset """
        (data, _, _), label = self.__readitem(idx)
        return data, label
 
    def __len__(self):
        return len(self.files)

    def sample(self, idx=None):
        """ Function for sampling dataset 
        
        Parameters:
            idx (int): sample idx
        Returns:
            (spectrogram_db, freqs, time)
        """
        if not idx:
            idx = random.uniform(0, len(self.files))
        return self.__readitem(idx)
        
