from enum import Enum
from torch.utils.data import Dataset
from data.sound import Sound


class SoundIndiciesDataset(Dataset, Sound):
    def __init__(self, filenames, hives, indicator_type):
        Sound.__init__(self, filenames, hives)
        self.indicator_type = indicator_type

    def __getitem__(self, idx):

        def get_ACI(sound_samples):
            pass

        def get_ADI(sound_samples):
            pass

        def get_AEI(sound_samples):
            pass

        def get_BI(sound_samples):
            pass

        # read sound samples from file
        sound_samples, sampling_rate, label = Sound.read_sound(self, idx)

        features = {
            self.SoundIndicator.ACI:    get_ACI,  
            self.SoundIndicator.ADI:    get_ADI, 
            self.SoundIndicator.AEI:    get_AEI,
            self.SoundIndicator.BI:    get_BI,
        }.get(self.indicator_type)(sound_samples)

        return (features, label)
        
    class SoundIndicator(Enum):
        ACI = 1     # acoustic complexity index
        ADI = 2     # acoustic diversity index
        AEI = 3     # acoustic evenness index
        BI = 4      # bioacustic index