import os
import torch
import random 
import numpy as np
import librosa

from math import sqrt
from torch.utils.data import Dataset
from scipy.io import wavfile
from sklearn.preprocessing import MinMaxScaler

class MfccDataset(Dataset):
    """ MFCC dataset - here we treat sound as stationary signal """
    def __init__(self, filenames, hives, mels, nfft, hop_len):
        self.files = filenames
        self.labels = hives
        self.n_mels = mels
        self.nfft = nfft
        self.hop_len = hop_len

    def __getitem__(self, idx):
        filename = self.files[idx]
        sampling_rate, sound_samples = wavfile.read(filename)
        hive_name = filename.split(os.sep)[-2].split("_")[0]
        label = next(index for index, name in enumerate(self.labels) if name == hive_name)
        if len(sound_samples.shape) > 1:
            # 2-channel recording
            sound_samples = sound_samples.T[0]
        sound_samples = sound_samples/(2.0**31)

        # calculate mfcc values
        mfccs = librosa.feature.mfcc(y=sound_samples, sr=sampling_rate, n_fft=self.nfft, hop_length=self.hop_len, n_mfcc=self.n_mels)
        mfccs = mfccs.astype(np.float32)
        mfccs_avg = np.mean(mfccs, axis=1)
        mfccs_avg = mfccs_avg[None, :]
        return (mfccs_avg, label)
        
    def __len__(self):
        return len(self.files)