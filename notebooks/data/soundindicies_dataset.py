from torch.utils.data import Dataset
from data.sound_dataset import SoundDataset

class SoundIndiciesDataset(Dataset, SoundDataset)
    def __init__(self, indicator_type)
        self.indicator_type = indicator_type

    class SoundIndicatotr(Enum):
        ACI = 1     # acoustic complexity index
        ADI = 2     # acoustic diversity index
        AEI = 3     # acoustic evenness index
        BI = 4      # bioacustic index