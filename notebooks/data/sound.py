import os
import numpy as np
from scipy.io import wavfile

def pcm2float(sig, dtype='float64'):
    """Convert PCM signal to floating point with a range from -1 to 1.

    Use dtype='float32' for single precision.

    Parameters
    ----------
    sig : array_like
        Input array, must have integral type.
    dtype : data type, optional
        Desired (floating point) data type.

    Returns
    -------
    numpy.ndarray
        Normalized floating point data.

    See Also
    --------
    float2pcm, dtype

    """
    sig = np.asarray(sig)
    if sig.dtype.kind not in 'iu':
        raise TypeError("'sig' must be an array of integers")
    dtype = np.dtype(dtype)
    if dtype.kind != 'f':
        raise TypeError("'dtype' must be a floating point type")

    i = np.iinfo(sig.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig.astype(dtype) - offset) / abs_max

def read_samples(filename, raw=False):
    """ Function for reading sound samples from wav file
    
    Paramters:
        filename (str): file to be read
        raw (bool): set to True if you don't want to scale samples (by max int32)

    Returns
        sound_samples (list(float)): scaled sound samples
        samplint_rate (int): sampling rate for audio
    """
    sampling_rate, sound_samples = wavfile.read(filename)

    if len(sound_samples.shape) > 1:
        # 2-channel recording
        sound_samples = sound_samples.T[0]

    if not raw:
        sound_samples = pcm2float(sound_samples, dtype='float64')

    return sound_samples, sampling_rate

class Sound():
    def __init__(self, filenames, labels):
        self.filenames = filenames
        self.labels = labels

    def read_sound(self, idx, raw=False):
        """ Method for reading sound
        
        Parameters:
            idx: idx of sound file to be read
            raw: if sound should be in raw format (dont convert from pcm to float)
            
        Returns:
            sounds_samples (list): list of sound samples
            sampling_rate (int): sampling rate
            label (int): label based on index from self.labels
         """
        filename = self.filenames[idx]
        sound_samples, sampling_rate = read_samples(filename, raw)
        hive_name = filename.split(os.sep)[-2].split("_")[0]
        try:
            label = next(index for index, name in enumerate(self.labels) if name == hive_name)
        except StopIteration as e:
            label = -1

        return sound_samples, sampling_rate, label
