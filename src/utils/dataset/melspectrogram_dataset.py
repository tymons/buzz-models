import os
import torch
import random 
import librosa
import numpy as np

from torch.utils.data import Dataset
from scipy.io import wavfile
from sklearn.preprocessing import MinMaxScaler

from utils.dataset.sound import Sound
from utils.data_utils import adjust_matrix, closest_power_2

class MelSpectrogramDataset(Dataset, Sound):
    """ MelSpectrogram dataset """
    def __init__(self, filenames, hives, nfft, hop_len, mels, truncate_power_two=False):
        """ Constructor for MelSepctrogram Dataset

        Parameters:
            filenames (list): list of strings with filenames
            hives (list): list of strings with hive names as it will server as lables
            nfft (int): how many samples for nfft
            hop_len (int): overlapping, samples for hop to next fft
            mels (int): mels
            truncate_power_two (bool): if we should truncate our shape to the nearest power of two
        """
        Sound.__init__(self, filenames, hives)
        self.nfft = nfft
        self.hop_len = hop_len
        self.mels = mels
        self.truncate = truncate_power_two

    def get_params(self):
        """ Function for returning params """
        return {
            'number_of_mels': self.mels,
            'nfft': self.nfft,
            'hop_len': self.hop_len
        }

    def __getitem__(self, idx):
        # read sound samples from file
        sound_samples, sampling_rate, label = Sound.read_sound(self, idx)

        mel = librosa.feature.melspectrogram(y=sound_samples, sr=sampling_rate, \
                                             n_fft=self.nfft, hop_length=self.hop_len, n_mels=self.mels)
        mel = librosa.power_to_db(mel, np.max)
        if self.truncate:
            mel = adjust_matrix(mel, 2**closest_power_2(mel.shape[0]), 2**closest_power_2(mel.shape[1]))

        initial_shape = mel.shape
        mel_scaled_spectrogram_db = MinMaxScaler().fit_transform(mel.reshape(-1, 1)).reshape((1, *initial_shape))
        mel_scaled_spectrogram_db = mel_scaled_spectrogram_db.astype(np.float32)

        return [mel_scaled_spectrogram_db], label
 
    def __len__(self):
        return len(self.filenames)

    def sample(self, idx=None):
        """ Function for sampling dataset 
        
        Parameters:
            idx (int): sample idx
        Returns:
            (spectrogram_db, freqs, time)
        """
        if not idx:
            idx = random.uniform(0, len(self.files))
        return self.__getitem__(idx)
        
