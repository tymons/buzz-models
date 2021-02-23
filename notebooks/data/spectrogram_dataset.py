import os
import torch
import random 
import librosa
import numpy as np

from torch.utils.data import Dataset
from scipy.io import wavfile
from sklearn.preprocessing import MinMaxScaler

class SpectrogramDataset(Dataset):
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
        self.files = filenames
        self.labels = hives
        self.nfft = nfft
        self.hop_len = hop_len
        self.fmax = fmax

    def __getitem__(self, idx):
        """ Function for getting item from Spectrogram dataset

        Parameters:
            idx (int): element idx

        Returns:
            ((spectrogram, frequencies, times), label) (tuple)
        """

        filename = self.files[idx]
        sample_rate, sound_samples = wavfile.read(filename)
        hive_name = filename.split(os.sep)[-2].split("_")[0]
        label = next(index for index, name in enumerate(self.labels) if name == hive_name) if self.labels else 0
        if len(sound_samples.shape) > 1:
            # 2-channel recording
            sound_samples = sound_samples.T[0]
        sound_samples = sound_samples/(2.0**31)

        # calculate spectrogram
        spectrogram = librosa.core.stft(sound_samples, n_fft=self.nfft, hop_length=self.hop_len)
        spectrogram_magnitude = np.abs(spectrogram)
        # spectrogram_phase = np.angle(spectrogram)
        spectrogram_db = librosa.amplitude_to_db(spectrogram_magnitude, ref=np.max)
        frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=self.nfft)
        times = (np.arange(0, spectrogram_magnitude.shape[1])*self.hop_len)/sample_rate
        if self.fmax:
            freq_slice = np.where((frequencies < self.fmax))
            frequencies = frequencies[freq_slice]
            spectrogram_db = spectrogram_db[freq_slice, :]

        initial_shape = spectrogram_db.shape
        scaled_spectrogram_db = MinMaxScaler().fit_transform(spectrogram_db.reshape(-1, 1)).reshape(initial_shape)
        scaled_spectrogram_db = scaled_spectrogram_db.astype(np.float32)
        return ((scaled_spectrogram_db, frequencies, times), label)
        
    def __len__(self):
        return len(self.files)