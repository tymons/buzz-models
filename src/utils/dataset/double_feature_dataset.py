from torch.utils.data import Dataset

class DoubleFeatureDataset(Dataset):
    """ Wrapper class for contrastive neural netowrks """
    def __init__(self, target, background, target_labels=[], background_labels=[],
                        base_class=None, **base_class_parameters):

        if isinstance(target, Dataset) and isinstance(background, Dataset):
            self.target = target
            self.background = target
        elif base_class is not None:
            self.target = base_class(target, target_labels, **base_class_parameters)
            self.background = base_class(background, background_labels, **base_class_parameters)
        else:
            raise ValueError("base class should be defined")

        assert len(target) == len(background)

    def __getitem__(self, idx):
        # function for returning target, background pair
        target_sample, label = self.target.__getitem__(idx)
        background_sample, _ = self.background.__getitem__(idx)
        return [*target_sample, *background_sample], label

    def __len__(self):
        return len(self.target) # as we assert len of target and backround at constructor