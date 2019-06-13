import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from python_speech_features import mfcc
from scipy.signal import windows
from scipy.fftpack import fft
from scipy.signal import spectrogram

import numpy as np


def calculate_mfcc(samples):
    sampling_rate = 3000
    mfcc_coefs = mfcc(signal=samples, samplerate=sampling_rate,
                      winlen=0.5, winstep=1, nfft=sampling_rate//2)   # Overlapping zeros
    # Return vector
    return np.ravel(mfcc_coefs)


class SmartulaSound:

    def __init__(self, timestamp, electromagnetic_field_on, samples=None, mfcc=None):
        if samples is not None:
            array = np.array(samples[0:1500]).astype(float)
            self.samples = array - array.mean()

        self.mfcc_feature_vector = calculate_mfcc(np.array(self.samples)) if mfcc is None else mfcc
        self.timestamp = timestamp
        self.electromagnetic_field_on = electromagnetic_field_on

    def get_fft(self):
        no_samples = len(self.samples)
        fs = no_samples  # We assume 1 sec samples
        period = 1 / fs

        x = np.linspace(0.0, no_samples * period, no_samples)
        sample_d = {'time [s]': x, 'value': self.samples}
        pdsamples = pd.DataFrame(sample_d, dtype=float)

        w = windows.hann(no_samples)
        yf = fft(self.samples * w)
        xf = np.linspace(0.0, 1.0 / (2.0 * period), no_samples // 2)
        freq_d = {'frequency': xf, 'amplitude real': (2 / no_samples * np.abs(yf[0:no_samples // 2]))}
        pdfreq = pd.DataFrame(freq_d, dtype=float)

        sns.set(style='darkgrid')
        fig, axs = plt.subplots(nrows=2)
        sns.lineplot(x='time [s]', y='value', data=pdsamples, ax=axs[0])
        sns.lineplot(x='frequency', y='amplitude real', data=pdfreq, ax=axs[1])
        plt.show()

        f, t, Sxx = spectrogram(self.samples, fs, window='hann')
        plt.pcolormesh(t, f, Sxx)
        plt.show()
