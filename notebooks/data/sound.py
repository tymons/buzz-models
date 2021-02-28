import os

from scipy.io import wavfile

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
        sound_samples = sound_samples/(2.0**31)

    return sound_samples, sampling_rate

class Sound():
    def __init__(self, filenames, labels):
        self.files = filenames
        self.labels = labels

    def read_sound(self, idx):
        """ Method for reading sound
        
        Parameters:
            idx: idx of sound file to be read

        Returns:
            sounds_samples (list): list of sound samples
            sampling_rate (int): sampling rate
            label (int): label based on index from self.labels
         """
        filename = self.files[idx]
        sound_samples, sampling_rate = read_samples(filename)
        hive_name = filename.split(os.sep)[-2].split("_")[0]
        try:
            label = next(index for index, name in enumerate(self.labels) if name == hive_name)
        except StopIteration as e:
            label = -1

        return sound_samples, sampling_rate, label
