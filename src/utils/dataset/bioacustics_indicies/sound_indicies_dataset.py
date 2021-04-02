import os
from enum import Enum

from utils.dataset.spectrogram_dataset import calculate_spectrogram
from utils.dataset.sound import read_samples
from utils.dataset.bioacustics_indicies.compute_indice import compute_ACI, compute_AEI, compute_BI, compute_spectrogram

from torch.utils.data import Dataset

class SoundIndiciesDataset(Dataset):
    def __init__(self, filenames, hives, indicator_type, nfft, hop_len, \
                    j_samples=None, adi_freq_step=None, scale_db=False):
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

        # aci related params
        self.nfft = nfft                        # nfft for spectrogram dataset
        self.hop_len = hop_len                  # hop len for spectrogram
        self.scale_db = scale_db                # if we should scale to our spectrogram
        self.j_samples = j_samples              # j samples for ACI indes

        # adi/ei related params
        self.freq_step = adi_freq_step          # freq step for ADI/AEI

    def get_params(self):
        """ Function for returning params """
        params = { 'indicator_type': self.indicator_type }
        if self.indicator_type == SoundIndicator.ACI:
            aci_params = {
                'aci_nfft': self.nfft,
                'aci_hop_len': self.hop_len,
                'scale_db': self.scale_db,
                'aci_j_samples': self.j_samples
            }
            params = {**params, **aci_params}
        elif self.indicator_type == SoundIndicator.AEI or self.indicator_type == SoundIndicator.ADI:
            adi_aei_params = {
                'freq_step': self.freq_step
            }
            params = {**params, **adi_aei_params}

        return params

    def __getitem__(self, idx):

        def get_ACI():
            """ Function for ACI value and temporal values calculation 
            
            Parameters needed for ACI calculation:
                nfft: number of samples per fft calculation
                hop_len: hop len for fft calculation
                j_samples: number of samples per j_bin
            Returns:
                tuple (ACI index, temporal ACIs)
            """
            assert self.j_samples is not None      

            sound_samples, sampling_rate = read_samples(filename)
            spectrogram_db, freqs, times = calculate_spectrogram(sound_samples, sampling_rate, nfft=self.nfft, \
                                                                    hop_len=self.hop_len, scale=False, db_scale=self.scale_db)
            spectrogram_db = spectrogram_db.squeeze()

            value, temporal_values = compute_ACI(spectrogram_db, j_bin=self.j_samples)

            return (value, temporal_values)

        def get_ADI():
            """ Function for ADI value calculation 
            
            Parameters needed for ADI calculation:
                adi_freq_step:

            Returns:
                ADI index (float)
            """
            assert self.freq_step is not none
            sound_samples, sampling_rate = read_samples(filename, raw=True)

            spectro, freqs = compute_spectrogram(sound_samples, sampling_rate, square=False)    # here we use numpy spectrogram implementation
                                                                                                # as librosa implementatin from calculate_spectrogram need floats
            max_freq = int((freqs[-1]+freqs[1]))
            value = compute_ADI(spectrogram, np.iinfo(sound_samples[0]).max, freq_band_Hz=max_freq/len(freqs), db_threshold=-50, \
                                    max_freq=max_freq, freq_step=self.freq_step)

            return value

        def get_AEI():
            """ Function for AEI value calculation 
            
            Parameters needed for AEI calculation:
                adi_freq_step:

            Returns:
                AEI index (float)
            """
            assert self.freq_step is not none
            
            sound_samples, sampling_rate = read_samples(filename, raw=True)
            spectro, freqs = compute_spectrogram(sound_samples, sampling_rate, square=False)    # here we use numpy spectrogram implementation
                                                                                                # as librosa implementatin from calculate_spectrogram need floats
            max_freq = int((freqs[-1]+freqs[1]))
            value = compute_AEI(spectrogram, np.iinfo(sound_samples[0]).max, freq_band_Hz=max_freq/len(freqs), db_threshold=-50, \
                                    max_freq=max_freq, freq_step=self.freq_step)

            return value

        def get_BI():
            """ Function for BI calculation 
            
            Parameters needed for BI calculation:
                None
            Returns:
                tuple (ADI index, temporal spectreogram BI mean)
            """
            sound_samples, sampling_rate = read_samples(filename, raw=True)
            spectro, freqs = compute_spectrogram(sound_samples, sampling_rate, square=False)    # here we use numpy spectrogram implementation
                                                                                                # as librosa implementatin from calculate_spectrogram need floats
            value, temporal_values = compute_BI(spectro, freqs, np.iinfo(sound_samples[0]).max, min_freq=20, max_freq=10000)

        # read sound samples from file
        filename = self.filenames[idx]
        hive_name = filename.split(os.sep)[-2].split("_")[0]
        try:
            label = next(index for index, name in enumerate(self.labels) if name == hive_name)
        except StopIteration as e:
            label = -1

        feature = {
            self.SoundIndicator.ACI:    get_ACI,  
            self.SoundIndicator.ADI:    get_ADI, 
            self.SoundIndicator.AEI:    get_AEI,
            self.SoundIndicator.BI:     get_BI,
        }.get(self.indicator_type)()

        return (feature, label)
    
    def __len__(self):
        return len(self.filenames)

    class SoundIndicator(Enum):
        ACI = 'aci'     # acoustic complexity index
        ADI = 'adi'     # acoustic diversity index
        AEI = 'aei'     # acoustic evenness index
        BI = 'bi'      # bioacustic index