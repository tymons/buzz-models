from torch.utils.data import Dataset, TensorDataset
from torchvision import transforms
from sklearn.preprocessing import MinMaxScaler

class CustomDataset(Dataset):
    """ Dataset with support of transforms. """
    
    def __init__(self, tensor_data, should_scale=False):
        self.data = tensor_data
        self.scale = should_scale

    def __getitem__(self, index):
        x = self.data[index]

        if self.scale:
            x -= x.min()
            x /= x.max()

        return x

    def __len__(self):
        return self.data.size(0)