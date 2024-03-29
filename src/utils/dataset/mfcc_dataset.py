import numpy as np
import librosa

from torch.utils.data import Dataset
from utils.dataset.sound import Sound

class MfccDataset(Dataset, Sound):
    """ MFCC dataset - here we treat sound as stationary signal by taking mean of melspectrogram """
    def __init__(self, filenames, hives, mels, nfft, hop_len):
        Sound.__init__(self, filenames, hives)
        self.n_mels = mels
        self.nfft = nfft
        self.hop_len = hop_len

    def get_params(self):
        """ Function for returning params """
        return {
            'number_of_mels': self.n_mels,
            'nfft': self.nfft,
            'hop_len': self.hop_len
        }

    def __getitem__(self, idx):
        # read sound samples and label
        sound_samples, sampling_rate, label = Sound.read_sound(self, idx)

        # calculate mfcc values
        mfccs = librosa.feature.mfcc(y=sound_samples, sr=sampling_rate, n_fft=self.nfft, hop_length=self.hop_len, n_mfcc=self.n_mels)
        mfccs = mfccs.astype(np.float32)
        mfccs_avg = np.mean(mfccs, axis=1)
        mfccs_avg = mfccs_avg[None, :]
        return [mfccs_avg], label
        
    def __len__(self):
        return len(self.filenames)