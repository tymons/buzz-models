from torch.utils.data import Dataset

class DoubleFeatureDataset(Dataset):
    """ Wrapper class for contrastive neural netowrks """
    def __init__(self, target_filenames, target_labels,
                        background_filenames, background_labels,
                        base_class, **base_class_parameters):

        assert len(target_filenames) == len(background_filenames)

        self.target = base_class(target_filenames, target_labels, **base_class_parameters)
        self.background = base_class(background_filenames, background_labels, **base_class_parameters)

    def __getitem__(self, idx):
        target_sample = self.target.__getitem__(idx)
        background_sample = self.background.__getitem__(idx)
        return (target_sample, background_sample)

    def __len__(self):
        return len(self.target) # as we assert len of target and backround at constructor