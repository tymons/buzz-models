import os
from enum import Enum

from data.spectrogram_dataset import calculate_spectrogram
from data.sound import read_samples
from data.indice.compute_indice import compute_ACI, compute_AEI, compute_BI, compute_spectrogram

from torch.utils.data import Dataset


class SoundIndiciesDataset(Dataset):
    def __init__(self, filenames, hives, indicator_type, n_fft, hop_len, \
                    j_samples=None, adi_freq_step=None):
        """
        Parameters:
            filenames (list(str)): list of filenames
            labels (list(str)): just labels
            indicator_type (SoundIndicator): type of the feature which will be returned by __getitem__ method
            nfft (int): number of samples per fft used in spectrum calculation
            hop_len (int): hop length for spectrum
            scale_to_db (bool): sclae to bool amplitude
            j_samples (int): samples per bin in ACI calculation

        """
        self.filenames = filenames              # sound filenames
        self.labels = hives                     # labels
        self.indicator_type = indicator_type    # ACI, ADI, AEI, see SoundIndicator class
        self.nfft = n_fft                       # nfft for spectrogram dataset
        self.hop_len = hop_len                  # hop len for spectrogram

        self.j_samples = j_samples              # j samples for ACI indes
        self.freq_step = adi_freq_step          # freq step for ADI/AEI

    def __getitem__(self, idx):

        def get_ACI(filename):
            """ Function for ACI value and temporal values calculation """
            assert self.j_samples is not None      

            sound_samples, sampling_rate = read_samples(filename)
            spectrogram_db, freqs, times = calculate_spectrogram(sound_samples, sampling_rate, nfft=self.nfft, \
                                                             hop_len=self.hop_len, scale=False, db_scale=True)
            spectrogram_db = spectrogram_db.squeeze()

            return compute_ACI(spectrogram_db, j_bin=self.j_samples)

        def get_ADI(filename):
            """ Function for ADI value calculation """
            assert self.freq_step is not none
            
            sound_samples, sampling_rate = read_samples(filename, raw=True)

            spectro, freqs = compute_spectrogram(sound_samples, sampling_rate, square=False)    # here we use numpy spectrogram implementation
                                                                                                # as librosa implementatin from calculate_spectrogram need floats
            max_freq = int((freqs[-1]+freqs[1]))
            value = compute_ADI(spectrogram, np.iinfo(sound_samples[0]).max, freq_band_Hz=max_freq/len(freqs), db_threshold=-50, \
                                    max_freq=max_freq, freq_step=self.freq_step)

            return (value, None)

        def get_AEI(sound_file):
            """ Function for AEI value calculation """
            assert self.freq_step is not none
            
            sound_samples, sampling_rate = read_samples(filename, raw=True)

            spectro, freqs = compute_spectrogram(sound_samples, sampling_rate, square=False)    # here we use numpy spectrogram implementation
                                                                                                # as librosa implementatin from calculate_spectrogram need floats
            max_freq = int((freqs[-1]+freqs[1]))
            value = compute_AEI(spectrogram, np.iinfo(sound_samples[0]).max, freq_band_Hz=max_freq/len(freqs), db_threshold=-50, \
                                    max_freq=max_freq, freq_step=self.freq_step)

            return (value, None)

        def get_BI(sound_file):
            """ Function for BI calculation """
            sound_samples, sampling_rate = read_samples(filename, raw=True)

            spectro, freqs = compute_spectrogram(sound_samples, sampling_rate, square=False)    # here we use numpy spectrogram implementation
                                                                                                # as librosa implementatin from calculate_spectrogram need floats
            return compute_BI(spectro, freqs, np.iinfo(sound_samples[0]).max, min_freq=20, max_freq=10000)

        # read sound samples from file
        filename = self.filenames[idx]
        hive_name = filename.split(os.sep)[-2].split("_")[0]
        try:
            label = next(index for index, name in enumerate(self.labels) if name == hive_name)
        except StopIteration as e:
            label = -1

        feature, features_temporal = {
            self.SoundIndicator.ACI:    get_ACI,  
            self.SoundIndicator.ADI:    get_ADI, 
            self.SoundIndicator.AEI:    get_AEI,
            self.SoundIndicator.BI:     get_BI,
        }.get(self.indicator_type)(filename)

        return ((feature, features_temporal), label)
        
    class SoundIndicator(Enum):
        ACI = 'aci'     # acoustic complexity index
        ADI = 'adi'     # acoustic diversity index
        AEI = 'aei'     # acoustic evenness index
        BI = 'bi'      # bioacustic index