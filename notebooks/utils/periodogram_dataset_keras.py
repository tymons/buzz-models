import os
import numpy as np

from tensorflow import keras
from scipy.fftpack import fft, fftfreq
from scipy.io import wavfile
from sklearn.preprocessing import MinMaxScaler

def read_sound_file(filename, slice_freq):
    """ Read and scales sound file """
    sample_rate, sound_samples = wavfile.read(filename)
    if len(sound_samples.shape) > 1:
            sound_samples = sound_samples.T[0]

    sound_samples = sound_samples/(2.0**31)
    periodogram = fft(sound_samples, n=sample_rate)
    periodogram = abs(periodogram[1:int(len(periodogram)/2)])

    if slice_freq:
        periodogram = periodogram[slice_freq[0]:slice_freq[1]]

    scaled_perio = MinMaxScaler().fit_transform(periodogram.reshape(-1, 1)).T
    scaled_perio = scaled_perio.squeeze()

    return scaled_perio
    
class PeriodogramGenerator(keras.utils.Sequence):
    def __init__(self, target_filenames, background_filenames,
                 labels, dim=(2048,), batch_size=32, slice_freq=None, shuffle=True):
        
        assert(len(target_filenames) == len(background_filenames))
        
        self.target_filenames = target_filenames
        self.background_filenames = background_filenames
        self.labels = labels
        self.dim = dim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.slice_freq = slice_freq
        
        self.on_epoch_end()
        
    def __len__(self):
        """Denotes the number of batches per epoch """
        return int(np.floor(len(self.target_filenames) / self.batch_size))
    
    def on_epoch_end(self):
        """ Updates indexes after each epoch """
        self.target_indexes = np.arange(len(self.target_filenames))
        self.background_indexes = np.arange(len(self.background_filenames))
        if self.shuffle == True:
            np.random.shuffle(self.target_indexes)
            np.random.shuffle(self.background_indexes)
            
    def __getitem__(self, index):
        """ Generate one batch of data """
        target_indexes = self.target_indexes[index*self.batch_size:(index+1)*self.batch_size]
        background_indexes = self.background_indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        target_filenames_temp = [self.target_filenames[k] for k in target_indexes]
        background_filenames_temp = [self.background_filenames[k] for k in background_indexes]
        
        target, background, _ = self.__data_generation(target_filenames_temp, background_filenames_temp)
        return [target, background]
    
    def __data_generation(self, target_filenames_temp, background_filenames_temp):
        """Generates data containing batch_size samples' # X  """
        # Initialization
        T = np.empty((self.batch_size, *self.dim))
        B = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        # Generate target data
        for i, filename in enumerate(target_filenames_temp):
            hive_name = filename.split(os.sep)[-2].split("_")[0]
            label = next(index for index, name in enumerate(self.labels) if name == hive_name)
                
            T[i, ] = read_sound_file(filename, self.slice_freq)
            y[i] = label
            
        # Generate background data
        for i, filename in enumerate(background_filenames_temp):
            B[i, ] = read_sound_file(filename, self.slice_freq)

        return T, B, y