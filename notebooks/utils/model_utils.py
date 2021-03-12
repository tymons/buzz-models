import torch
import matplotlib.pyplot as plt
import numpy as np

from torch.utils import data as tdata
from torch import nn


def train_model(model, model_params, patience,
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
