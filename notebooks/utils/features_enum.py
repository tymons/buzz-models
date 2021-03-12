from enum import Enum

from torch.utils.data import DataLoader, random_split

from data.periodogram_dataset import PeriodogramDataset
from data.spectrogram_dataset import SpectrogramDataset
from data.melspectrogram_dataset import MelSpectrogramDataset
from data.indice.sound_indicies_dataset import SoundIndiciesDataset


class SoundFeatureDataset(Enum):
    SPECTROGRAM = 'spectrogram'
    MELSPECTROGRAM = 'melspectrogram'
    PERIODOGRAM = 'periodogram'
    MFCC = 'mfcc'
    SOUND_INDICIES = 'indicies' # bioacustic signal indicies


    def _get_spectrogram_dataset(sound_filenames, labels, **feature_parameters):
        """ Function for getting spectrogram """
        nfft = feature_parameters.get('nfft', 4096)
        hop_len = feature_parameters.get('hop_len', (4096//3)+30)
        fmax = feature_parameters.get('fmax', 2750)

        return SpectrogramDataset(sound_filenames, labels, nfft=nfft, hop_len=hop_len, fmax=fmax)

    def _get_melspectrogram_dataset(sound_filenames, labels, **feature_parameters):
        """ Function for getting melspectrogram dataset """
        nfft = feature_parameters.get('nfft', 4096)
        hop_len = feature_parameters.get('hop_len', (4096//3)+30)
        fmax = feature_parameters.get('fmax', 2750)
        no_mels = feature_parameters.get('mels', 64)

        return MelSpectrogramDataset(sound_filenames, labels, nfft=nfft, hop_len=hop_len, mels=no_mels)

    def _get_periodogram_dataset(sound_filenames, labels, **feature_parameters):
        """ Function for getting periodogram dataset """
        slice_freq = feature_parameters.get('slice_freq', (0, 2048))

        return PeriodogramDataset(sound_filenames, lables, slice_freq=slice_freq)

    def _get_mfcc_dataset(sound_filenames, labels, **feature_parameters):
        """ Function for getting mfcc from sound """
        nfft = feature_parameters.get('nfft', 4096)
        hop_len = feature_parameters.get('hop_len', (4096//3)+30)
        fmax = feature_parameters.get('fmax', 2750)
        no_mels = feature_parameters.get('mels', 64)

        return MfccDataset(sound_filenames, labels, nfft=nfft, hop_len=hop_len, mels=no_mels)

    def _get_indicies_dataset(sound_filenames, labels, **feature_parameters):
        """ Function for getting indicies from sounds """
        indicator_type = feature_parameters.get('indicator_type', 'aci')

        return SoundIndiciesDataset(sound_filenames, labels, SoundIndiciesDataset.SoundIndicator(indicator_type), **feature_parameters)

    @classmethod
    def get_dataloaders(cls, input_type, sound_filenames, labels, batch_size, \
                            ratio=0.15, num_workers=4, **feature_parameters):
        """ Function for getting dataloaders 
        
        Parameters:
            input_type (str): input type, should be one oof InputType Enum values
            sound_filenames (list(str)): list with sound filenames
            labels (list(str)): label names
            batch_size (int): batch size for dataloader
            ratio (int): ratio between train dataset and validation dataset
            num_workers (int): num workers for dataloaders

        Returns:
            train_loader, val_loader (tuple(Dataloader, Dataloader)): train dataloader, validation dataloder
        """
        method_name = f'_get_{input_type.lower()}_dataset'
        function = getattr(cls, method_name, lambda: 'invalid dataset')
        dataset = function(sound_filenames, labels, **feature_parameters)

        val_amount = int(dataset.__len__() * ratio)
        train_set, val_set = random_split(dataset, [(dataset.__len__() - val_amount), val_amount])
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        return train_loader, val_loader