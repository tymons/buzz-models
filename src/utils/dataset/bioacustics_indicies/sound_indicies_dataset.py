import os
import numpy as np
from enum import Enum

from .. import sound
from ..spectrogram_dataset import calculate_spectrogram
from . import compute_indice as indice

from torch.utils.data import Dataset

class SoundIndiciesDataset(Dataset, sound.Sound):
    def __init__(self, filenames, hives, indicator_type, nfft, hop_len=512, \
                    aci_j_samples=None, adi_freq_step=None, scale_db=False, 
                    ndsi_anthrophony=(10, 250), ndsi_biophony=(250, 3000), db_threshold=-50):
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
        sound.Sound.__init__(self, filenames, hives)

        self.filenames = filenames              # sound filenames
        self.labels = hives                     # labels
        self.indicator_type = indicator_type    # ACI, ADI, AEI, see SoundIndicator class

        # common parameters
        self.nfft = nfft                        # nfft for spectrogram dataset
        self.hop_len = hop_len                  # hop len for spectrogram
        self.scale_db = scale_db                # if we should scale to our spectrogram

        # aci related params
        self.j_samples = aci_j_samples          # j samples for ACI indes

        # adi/ei related params
        self.freq_step = adi_freq_step          # freq step for ADI/AEI
        self.db_threshold = db_threshold        # threshold for ADI/AEI

        # ndsi related params
        self.ndsi_anthrophony = ndsi_anthrophony if type(ndsi_anthrophony) == tuple else tuple(ndsi_anthrophony)
        self.ndsi_biophony = ndsi_biophony if type(ndsi_biophony) == tuple else tuple(ndsi_biophony)

    def get_params(self):
        """ Function for returning params """
        params = { 'indicator_type': self.indicator_type }
        if self.indicator_type == self.SoundIndicator.ACI:
            aci_params = {
                'aci_nfft': self.nfft,
                'aci_hop_len': self.hop_len,
                'scale_db': self.scale_db,
                'aci_j_samples': self.j_samples
            }
            params = {**params, **aci_params}
        elif self.indicator_type == self.SoundIndicator.AEI or self.indicator_type == self.SoundIndicator.ADI:
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

            sound_samples, sampling_rate, label = sound.Sound.read_sound(self, idx, raw=False)
            spectrogram_db, _, _ = calculate_spectrogram(sound_samples, sampling_rate, nfft=self.nfft, \
                                                                    hop_len=self.hop_len, scale=False, db_scale=self.scale_db)
            spectrogram_db = spectrogram_db.squeeze()
            
            value, temporal_values = indice.compute_ACI(spectrogram_db, j_bin=self.j_samples)

            return (value, temporal_values), label

        def get_ADI():
            """ Function for ADI value calculation 
            
            Parameters needed for ADI calculation:
                adi_freq_step:

            Returns:
                ADI index (float)
            """
            assert self.freq_step is not None
            sound_samples, sampling_rate, label =  sound.Sound.read_sound(self, idx, raw=True)
            spectrogram, freqs = indice.compute_spectrogram(sound_samples, sampling_rate, square=True,
                                    windowLength=self.nfft, windowHop=self.hop_len, db_scale=True) # here we use numpy spectrogram implementation
                                                                                                   # as librosa implementatin from calculate_spectrogram need floats
            max_freq = int((freqs[-1]+freqs[1]))
            value = indice.compute_ADI(spectrogram, np.iinfo(sound_samples[0]).max, freq_band_Hz=max_freq/len(freqs), dbfs_threshold=self.db_threshold, \
                                    max_freq=max_freq, freq_step=self.freq_step)

            return (value, None), label

        def get_AEI():
            """ Function for AEI value calculation 
            
            Parameters needed for AEI calculation:
                adi_freq_step:

            Returns:
                AEI index (float)
            """
            assert self.freq_step is not None
            sound_samples, sampling_rate, label =  sound.Sound.read_sound(self, idx, raw=True)

            spectrogram, freqs = indice.compute_spectrogram(sound_samples, sampling_rate, square=True,
                                    windowLength=self.nfft, windowHop=self.hop_len, db_scale=True) # here we use numpy spectrogram implementation
                                                                                                            # as librosa implementatin from calculate_spectrogram need floats
            max_freq = int((freqs[-1]+freqs[1]))
            value = indice.compute_AEI(spectrogram, np.iinfo(sound_samples[0]).max, freq_band_Hz=max_freq/len(freqs), dbfs_threshold=self.db_threshold, \
                                    max_freq=max_freq, freq_step=self.freq_step)

            return (value, None), label

        def get_BI():
            """ Function for BI calculation 
            
            Parameters needed for BI calculation:
                None
            Returns:
                tuple (ADI index, temporal spectreogram BI mean)
            """
            sound_samples, sampling_rate, label =  sound.Sound.read_sound(self, idx, raw=True)

            spectro, freqs = indice.compute_spectrogram(sound_samples, sampling_rate, windowLength=self.nfft, windowHop=self.hop_len, db_scale=self.scale_db, square=True)    
                                                                                                # here we use numpy spectrogram implementation
                                                                                                # as librosa implementatin from calculate_spectrogram need floats
            value, temporal_values = indice.compute_BI(spectro, freqs, np.iinfo(sound_samples[0]).max, min_freq=10, max_freq=5000)

            return (value, temporal_values), label
        
        def get_H():
            """ Function for entropy calculation """
            sound_samples, sampling_rate, label =  sound.Sound.read_sound(self, idx, raw=False)
            spectro, _, _ = calculate_spectrogram(sound_samples, sampling_rate, nfft=self.nfft, hop_len=self.hop_len, scale=False, db_scale=self.scale_db)
            spectro = spectro.squeeze()
            value = indice.compute_SH(spectro)

            return (value, None), label

        def get_NDSI():
            """ Function for NDSI calculation """
            sound_samples, sampling_rate, label =  sound.Sound.read_sound(self, idx, raw=False)
            value = indice.compute_NDSI(sound_samples, sampling_rate, windowLength=self.nfft, anthrophony=self.ndsi_anthrophony, biophony=self.ndsi_biophony)
            return (value, None), label

        def get_ZCR():
            sound_samples, _, label =  sound.Sound.read_sound(self, idx, raw=True)
            zcr_frames = indice.compute_zcr(sound_samples, windowLength=self.nfft, windowHop=self.hop_len)
            return (np.mean(zcr_frames), zcr_frames), label

        feature, label = {
            self.SoundIndicator.ACI:    get_ACI,  
            self.SoundIndicator.ADI:    get_ADI, 
            self.SoundIndicator.AEI:    get_AEI,
            self.SoundIndicator.BI:     get_BI,
            self.SoundIndicator.H:      get_H,
            self.SoundIndicator.NDSI:   get_NDSI,
            self.SoundIndicator.ZCR:    get_ZCR,
        }.get(self.indicator_type)()

        return [feature], label
    
    def __len__(self):
        return len(self.filenames)

    class SoundIndicator(Enum):
        ACI = 'aci'     # acoustic complexity index
        ADI = 'adi'     # acoustic diversity index
        AEI = 'aei'     # acoustic evenness index
        BI = 'bi'       # bioacustic index
        H = 'entropy'   # temporal entropy
        NDSI = 'ndsi'   # normalized difference soundscape index
        ZCR = 'zcr'     # zero crossing rate