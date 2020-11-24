import torch
import matplotlib.pyplot as plt
import numpy as np

from torch.utils import data as tdata
from torch import nn


class BasicAutoencoder(nn.Module):
    def __init__(self):
        super(BasicAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=256, out_features=128),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(True),
            nn.Linear(in_features=128, out_features=64),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(True),
            nn.Linear(in_features=64, out_features=32),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(in_features=32, out_features=64),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(True),
            nn.Linear(in_features=64, out_features=128),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(True),
            nn.Linear(in_features=128, out_features=256),
            nn.BatchNorm1d(num_features=256),
            nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class View(nn.Module):
    """ Function for nn.Sequentional to reshape data """

    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


def _conv2d_block(in_f, out_f, *args, **kwargs):
    """ Function for building convolutional block

        Attributes
            in_f - number of input features
            out_f - number of output features
    """
    return nn.Sequential(
        nn.Conv2d(in_f, out_f, *args, **kwargs),
        nn.BatchNorm2d(out_f),
        nn.ReLU(),
        nn.Dropout2d(p=0.2)
    )


def _conv2d_transpose_block(in_f, out_f, *args, **kwargs):
    """ Function for building transpose convolutional block

        Attributes
            in_f - number of input features
            out_f - number of output features
    """
    return nn.Sequential(
        nn.ConvTranspose2d(in_f, out_f, *args, **kwargs),
        nn.BatchNorm2d(out_f),
        nn.ReLU(),
        nn.Dropout2d(p=0.2)
    )


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            # [1x256x64] => [64x256x64]
            _conv2d_block(1, 128, kernel_size=3, padding=1),
            # [64x256x64] => [64x128x32]
            nn.MaxPool2d(2, 2),
            # [64x128x32] => [32x128x32]
            _conv2d_block(128, 64, kernel_size=3, padding=1),
            # [32x128x32] => [32x64x16]
            nn.MaxPool2d(2, 2),
            # [32x64x16] => [16x64x16]
            _conv2d_block(64, 32, kernel_size=3, padding=1),
            # [16x64x16] => [16x32x8]
            nn.MaxPool2d(2, 2),
            # [16x32x8] => [4x32x8]
            _conv2d_block(32, 16, kernel_size=3, padding=1),
            # [4x32x8] => [4x16x4]
            nn.MaxPool2d(2, 2),
            # [4x16x4] => [1x256]
            nn.Flatten(),
            # [1x256] => [1x64]
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            # [1x64] => [1x256]
            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            # [1x256] => [4x16x4]
            View([-1, 16, 16, 4]),
            # [4x16x4] => [16x32x8]
            _conv2d_transpose_block(16, 32, kernel_size=2, stride=2),
            # [16x32x8] => [32x64x16]
            _conv2d_transpose_block(32, 64, kernel_size=2, stride=2),
            # [32x64x16] => [64x128x32]
            _conv2d_transpose_block(64, 128, kernel_size=2, stride=2),
            # [64x128x32] => [1x256x64]
            nn.ConvTranspose2d(128, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x


def train_model(model, learning_rate, weight_decay, num_epochs, patience,
                dataloader_train, dataloader_val, checkpoint_name='checkpoint.pth'):
    """
    Function for training model

    attribute: model - model which should be trained
    attribute: learning_rate - learning rate for the model
    attribute: weight_decay - weight decay for learning
    attribute: num_epochs - number epochs
    attribute: patience - patience for early stopping
    attribute: dataloader_train - train data loader
    attribute: dataloader_val - validation data loader
    attribute: checkpoint_name - checkpoint name for early stopping
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'model training performed on {device}')

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # monitor training loss per batch
    train_loss = []
    # monitor validation loss per batch
    val_loss = []
    # save avg train losses for early stopping visualization
    avg_train_loss = []
    # save avg train losses for early stopping visualization
    avg_val_loss = []
    # counter for patience in early sotpping
    patience_counter = 0
    # best validation score
    best_val_loss = -1
    # model checkpoint filename
    checkpoint_filename = checkpoint_name
    # early stopping epoch
    win_epoch = 0

    # pass model to gpu if is available
    model.to(device)

    for epoch in range(1, num_epochs + 1):
        ###################
        # train the model #
        ###################
        model.train()
        for data in dataloader_train:
            # transfer data to device
            input_data = data[0].to(device)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass
            outputs = model(input_data)
            # calculate the loss
            loss = criterion(outputs, input_data)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            train_loss.append(loss.item())

        ###################
        # val the model   #
        ###################
        model.eval()
        for val_data in dataloader_val:
            # transfer data to device
            input_data_val = val_data[0].to(device)
            # forward pass
            val_outputs = model(input_data_val)
            # calculate the loss
            vloss = criterion(val_outputs, input_data_val)
            # update running val loss
            val_loss.append(vloss.item())

        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_loss)
        val_loss = np.average(val_loss)
        avg_train_loss.append(train_loss)
        avg_val_loss.append(val_loss)

        epoch_len = len(str(num_epochs))
        # print avg training statistics
        print(
            f'[{epoch:>{epoch_len}}/{num_epochs:>{epoch_len}}] train_loss: {train_loss:.5f} valid_loss: {val_loss:.5f}',
            end=' ', flush=True)

        if val_loss < best_val_loss or best_val_loss == -1:
            # new checkpoint
            print("checkpoint!")
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_filename)
            win_epoch = epoch
        elif patience_counter >= patience:
            print("early stopping.")
            print(f"=> loading checkpoint {checkpoint_filename}")
            device = torch.device("cuda")
            model.load_state_dict(torch.load(checkpoint_filename))
            break
        else:
            print(".")
            patience_counter = patience_counter + 1

        # clear batch losses
        train_loss = []
        val_loss = []

    fig = plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, epoch + 1), avg_train_loss, 'r', label="train loss")
    plt.plot(np.arange(1, epoch + 1), avg_val_loss, 'b', label="validation loss")
    plt.axvline(win_epoch, linestyle='--', color='g', label='Early Stopping Checkpoint')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

    return model


def conv2d_encode(model, data_input):
    """ Function for encoding data and returning encoded """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_tensor = torch.Tensor(data_input)
    dataset_tensor = dataset_tensor[:, None, :, :]
    dataset_tensor = tdata.TensorDataset(dataset_tensor)
    dataset = tdata.DataLoader(dataset_tensor, batch_size=32, shuffle=True)
    encoded_data = []

    model.eval()
    with torch.no_grad():
        for data in dataset:
            periodograms = data[0].to(device)
            output = model.encoder(periodograms).cpu().numpy().squeeze()
            encoded_data.extend(output)

    return encoded_data


def basic_ae_encode(model, data_input):
    """ Function for encoding with autoencoder """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_tensor = torch.Tensor(data_input)
    dataset_tensor = tdata.TensorDataset(dataset_tensor)
    dataset = tdata.DataLoader(dataset_tensor, batch_size=32, shuffle=True)
    encoded_data = []

    model.eval()
    with torch.no_grad():
        for data in dataset:
            periodograms = data[0].to(device)
            output = model.encoder(periodograms).cpu().numpy().squeeze()
            encoded_data.extend(output)

    return encoded_data
