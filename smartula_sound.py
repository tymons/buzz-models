import numpy as np
import seaborn as sns
from sklearn import preprocessing
from scipy.signal import windows
import matplotlib.pyplot as plt
import pandas as pd

from scipy.fftpack import fft


class SmartulaSound:

    def __init__(self, samples):
        self.samples_normalized = np.reshape(preprocessing.normalize([samples]), -1)
        array = np.array(samples).astype(float)
        self.samples = array - array.mean()

    def get_fft(self):
        no_samples = len(self.samples)
        period = 1 / no_samples  # We assume 1 sec samples

        x = np.linspace(0.0, no_samples * period, no_samples)
        sample_d = {'time [s]': x, 'value': self.samples}
        pdsamples = pd.DataFrame(sample_d, dtype=float)

        w = windows.blackman(no_samples)
        yf = fft(self.samples*w)
        xf = np.linspace(0.0, 1.0 / (2.0 * period), no_samples // 2)

        # N = 600
        # T = 1.0 / 800.0
        # x = np.linspace(0.0, N * T, N)
        # y = np.sin(50.0 * 2.0 * np.pi * x) + 0.5 * np.sin(80.0 * 2.0 * np.pi * x)
        # yf = fft(y)
        # xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
        plt.plot(xf, np.abs(yf[0:no_samples//2]))
        plt.show()

        # freq_d = {'frequency': xf, 'amplitude real': (2/no_samples * np.abs(yf[0:no_samples//2]))}
        # pdfreq = pd.DataFrame(freq_d, dtype=float)
        #
        # sns.set(style='darkgrid')
        # fig, axs = plt.subplots(nrows=2)
#        sns.lineplot(x='time [s]', y='value', data=pdsamples, ax=axs[0])
#        sns.lineplot(x='frequency', y='amplitude real', data=pdfreq, ax=axs[1])
