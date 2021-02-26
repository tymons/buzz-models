import os

from scipy.io import wavfile

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
        sampling_rate, sound_samples = wavfile.read(filename)
        hive_name = filename.split(os.sep)[-2].split("_")[0]
        label = next(index for index, name in enumerate(self.labels) if name == hive_name)
        if len(sound_samples.shape) > 1:
            # 2-channel recording
            sound_samples = sound_samples.T[0]
        sound_samples = sound_samples/(2.0**31)

        return sound_samples, sampling_rate, label
