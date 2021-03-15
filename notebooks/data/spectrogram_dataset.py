import os
import torch
import random 
import librosa
import numpy as np

from torch.utils.data import Dataset
from scipy.io import wavfile
from sklearn.preprocessing import MinMaxScaler

from data.sound import Sound

def calculate_spectrogram(samples, sampling_rate, nfft, hop_len, fmax=None, scale=True, db_scale=True):
    """ function for calculating spectrogram 
    
    Params:
        samples (list(float)): audio samples from which spectrogram should be calculated
        sampling_rate (int): sampling rate for audio
        nfft (int): samples for fft calculation
        hop_len (inst): samples for hop (next fft calculation)
        fmax (int): max frequency for calculated spectrogram to be constrained
        scale (bool): should scale spectrogram between 0 and 1
        db_scale (bool): should magnitued be converted to db

    Returns
        spectrogram_magnitude (list): spectrogram
    """
    # calculate spectrogram
    spectrogram = librosa.core.stft(samples, n_fft=nfft, hop_length=hop_len)
    spectrogram_magnitude = np.abs(spectrogram)
    # spectrogram_phase = np.angle(spectrogram)
    if db_scale:
        spectrogram_magnitude = librosa.amplitude_to_db(spectrogram_magnitude, ref=np.max)

    frequencies = librosa.fft_frequencies(sr=sampling_rate, n_fft=nfft)
    times = (np.arange(0, spectrogram_magnitude.shape[1])*hop_len)/sampling_rate

    if fmax:
        freq_slice = np.where((frequencies < fmax))
        frequencies = frequencies[freq_slice]
        spectrogram_magnitude = spectrogram_magnitude[freq_slice, :]      # here extra dimension will be added
    else:
        spectrogram_magnitude = spectrogram_magnitude[None, :, :]         # but without indicies we should add it manually

    if scale:
        initial_shape = spectrogram_magnitude.shape
        spectrogram_magnitude = MinMaxScaler().fit_transform(spectrogram_magnitude.reshape(-1, 1)).reshape(initial_shape)
    
    spectrogram_magnitude = spectrogram_magnitude.astype(np.float32)
    return spectrogram_magnitude, frequencies, times

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

    def read_item(self, idx):
        """ Function for getting item from Spectrogram dataset

        Parameters:
            idx (int): element idx

        Returns:
            ((spectrogram, frequencies, times), label) (tuple)
        """
        sound_samples, sampling_rate, label = Sound.read_sound(self, idx)
        spectrogram_db, frequencies, times = calculate_spectrogram(sound_samples, sampling_rate, self.nfft, \
                                                                     self.hop_len, self.fmax, scale=True)

        return ((spectrogram_db, frequencies, times), label)

    def get_item(self, idx):
        """ Wrapper for reading spectrogram from file along with label """
        (data, _, _), labels = self.read_item(idx)
        return data, labels

    def __getitem__(self, idx):
        """ Wrapper for getting item from Spectrogram dataset """
        (data, _, _), labels = self.read_item(idx)
        return data, labels
 
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
            idx = int(random.uniform(0, len(self.files)))
        return self.read_item(idx)
        
