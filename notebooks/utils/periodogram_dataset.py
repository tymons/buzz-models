import os

from math import sqrt
from torch.utils.data import IterableDataset
from scipy.fftpack import fft, fftfreq
from scipy.io import wavfile

class PeriodogramDataset(IterableDataset):
    """ Periodogram dataset """
    def __init__(self, filenames, hives, slice_freq=None):
        self.files = filenames
        self.labels = hives
        self._index = 0
        self.slice_freq = slice_freq

    def __iter__(self):
        return self
    
    def __next__(self):
        if self._index < len(self.files):
            filename = self.files[self._index]
            sample_rate, sound_samples = wavfile.read(filename)
            hive_name = filename.split(os.sep)[-2].split("_")[0]
            label = -1
            try:
                label = next(idx for idx, name in enumerate(self.labels) if name == hive_name)
            except StopIteration as e:
                pass
        
            self._index += 1
            sound_samples = sound_samples.T[0]/(2.0**31)
            rms = sqrt(sum(sound_samples**2)/len(sound_samples))
            if rms < 0.7:
                periodogram = fft(sound_samples, n=sample_rate)
                periodogram = abs(periodogram[1:int(len(periodogram)/2)])
                if self.slice_freq:
                    periodogram = periodogram[self.slice_freq[0]:self.slice_freq[1]]
                return periodogram, label
            else:
                self.__next__()
        raise StopIteration
        
