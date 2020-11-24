import torch
import collections
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from torch.utils import data as tdata
from sklearn.svm import SVC
from datetime import datetime, timedelta


def read_sensor_data(filename, hive_sn, hives_ids, start_time, end_time, timezone_offset_hours, sensor_column_name):
    """ Function for reading smartula sensor file (from grafana) and build pandas dataframe """
    df_sensor_data = pd.read_csv(filename, skiprows=1, sep=";")

    if hive_sn not in hives_ids:
        print(f"Hive {hive_sn} is not in hives_ids set! Returning empty dataframe")
        return pd.DataFrame()

    # change series column to be coherent with sounds
    for hive in hives_ids:
        df_sensor_data.loc[df_sensor_data['Series'].str.contains(hive[2:]), 'Series'] = hive

    # change column names to match sound
    df_sensor_data.columns = ['name', 'datetime', sensor_column_name]
    # convert timestamp to pandas timestamp
    df_sensor_data['datetime'] = [(datetime.strptime(date_pd[:-6], '%Y-%m-%dT%H:%M:%S') +
                                   timedelta(hours=timezone_offset_hours)) for date_pd in
                                  df_sensor_data['datetime'].values.tolist()]

    df_sensor_data = df_sensor_data[(df_sensor_data['name'] == hive_sn) & (df_sensor_data['datetime'] > start_time) & (
            df_sensor_data['datetime'] < end_time)]
    df_sensor_data.set_index('datetime', inplace=True)
    print(f"got {df_sensor_data[sensor_column_name].count()} of {sensor_column_name} samples")

    return df_sensor_data


def merge_dataframes_ontimestamp(df_merge_to, *args):
    """ Merging dataframes to df_merge_to """
    df_hive_data_ua = df_merge_to
    for dataframe in args:
        df_hive_data_ua = pd.merge(df_hive_data_ua, dataframe.reindex(df_hive_data_ua.index, method='nearest'),
                                   on=['datetime', 'name'])

    return df_hive_data_ua


def prepare_dataset1d(data_df, train_ratio):
    """ Function for preparing dataset for autoencoder

        attributes: data_df - pandas dataframe column
        attributes: train_ratio - radio of train set size
        return train_dataset, test_dataset
    """
    train_data_size = int(data_df.shape[0] * train_ratio)
    val_data_size = data_df.shape[0] - train_data_size

    dataset_tensor = torch.Tensor(data_df.values.tolist())
    print(f"Dataset shape: {dataset_tensor.shape}")
    print(f"Train set size: {train_data_size}")
    print(f"Validation set size: {val_data_size}")

    # add one extra dimension as it is required for conv layer
    # dataset_tensor = dataset_tensor[:, None, :]
    dataset = tdata.TensorDataset(dataset_tensor)
    train_set, val_set = tdata.random_split(dataset, [train_data_size, val_data_size])

    return train_set, val_set


def prepare_dataset2d(data_df, train_ratio):
    """ Function for preparing dataset for autoencoder

        attributes: data_df - pandas dataframe column
        attributes: train_ratio - radio of train set size
        return train_dataset, test_dataset
    """
    train_data_size = int(data_df.shape[0] * train_ratio)
    val_data_size = data_df.shape[0] - train_data_size

    dataset_tensor = torch.Tensor(data_df.values.tolist())
    print(f"Dataset shape: {dataset_tensor.shape}")
    print(f"Train set size: {train_data_size}")
    print(f"Validation set size: {val_data_size}")

    # add one extra dimension as it is required for conv layer
    new_dataset = dataset_tensor[:, None, :, :]
    dataset = tdata.TensorDataset(new_dataset)
    train_set, val_set = tdata.random_split(dataset, [train_data_size, val_data_size])

    return train_set, val_set


def flatten(x):
    """
    Flatten array
    :param x:
    :return:
    """
    if isinstance(x, collections.abc.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]


def merge_columns(data_frame, column_names):
    """ Function for merging columns with irregular size """
    return [flatten(val) for val in data_frame[column_names].values.tolist()]


def plot_spectrogram(frequency, time_x, spectrocgram, title):
    fig = plt.figure(figsize=(6, 4))
    plt.title(title)
    plt.pcolormesh(time_x, frequency, spectrocgram)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()


def plot_hour_shift(*args, labels_list, xticklabels):
    """ Function for plotting n-hour shift """
    fig, axs = plt.subplots(len(args[0]) // 2, 2, figsize=(10, 8))
    fig.subplots_adjust(hspace=0.7)

    colors = ['ro', 'bx', 'go', 'yx', 'ko']

    if len(args) > len(colors):
        print('warning your accuracies are bigger than colors for plot!')

    for feature_idx, accuracy in enumerate(args):
        for acc_idx, acc_in_shift in enumerate(accuracy):
            axs[acc_idx // 2][acc_idx % 2].plot(acc_in_shift, colors[feature_idx], label=labels_list[feature_idx])
            axs[acc_idx // 2][acc_idx % 2].grid()
            axs[acc_idx // 2][acc_idx % 2].set_xticks(np.arange(0, len(xticklabels), 1))
            axs[acc_idx // 2][acc_idx % 2].tick_params(axis='x', rotation=270)
            axs[acc_idx // 2][acc_idx % 2].set_xticklabels(xticklabels)
            axs[acc_idx // 2][acc_idx % 2].set_title(f'{acc_idx + 1} hour long bee-night')
            axs[acc_idx // 2][acc_idx % 2].set_ylabel('SVM accuracy')
            axs[acc_idx // 2][acc_idx % 2].set_xlabel('Hour')
            handles, labels = axs[acc_idx // 2][acc_idx % 2].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper right')

    fig.show()


def search_best_night_day(input_data, feature_name, days_as_test, start_hours, max_shift, verbose=0):
    """ Function performing One-class SVM

        attribute: train_data - pandas series dataframe
        attribute: feature_name - name of column from dataframe which will be used as feature
        attribute: days_test - number of last days which will be used to create train data
        attribute: start_hours - list with start hours
        attribute: max_shift - max shift in hours
    """
    max_accuracy = 0

    accs_per_shift = []
    final_accs = []

    for shift in range(1, max_shift + 1):
        for start_hour in start_hours:
            data_to_svm = pd.DataFrame(input_data)
            data_to_svm.sort_index(inplace=True)

            end_hour = (start_hour + shift) % 24
            if end_hour > 12 or start_hour < max_shift:
                data_to_svm['is_night'] = (data_to_svm.index.hour >= start_hour) & (data_to_svm.index.hour <= end_hour)
            else:
                data_to_svm['is_night'] = (data_to_svm.index.hour >= start_hour) | (data_to_svm.index.hour <= end_hour)

            samples_in_day = data_to_svm[data_to_svm.index < (data_to_svm.index[0] + timedelta(days=1))].count()
            data_test = data_to_svm.tail(samples_in_day[0] * days_as_test)
            data_train = data_to_svm[~data_to_svm.isin(data_test)].dropna(how='all')

            train_data = data_train[feature_name].values.tolist()
            train_labels = data_train['is_night'].values.tolist()
            test_data = data_test[feature_name].values.tolist()
            test_labels = data_test['is_night'].values.tolist()

            if verbose > 0:
                print(f'learning with train data size: {len(train_data)} and test data size: {len(test_data)}')
                print(f'number of nights in train/test data: {sum(train_labels)}/{sum(test_labels)}')
            svc = SVC(kernel='rbf', class_weight='balanced', gamma='auto')
            svc.fit(train_data, train_labels)
            predicted = svc.predict(test_data)

            sum_correct = 0
            for idx, label_predicted in enumerate(predicted):
                if (label_predicted == int(test_labels[idx])):
                    sum_correct += 1

            accuracy = (sum_correct / len(test_labels) * 100)
            if accuracy > max_accuracy:
                if verbose > 0:
                    print(f'new max acuuracy for {start_hour} to {end_hour}, accuracy: {accuracy:.2f}')
                max_accuracy = accuracy

            if verbose > 0:
                print(f'for night start at {start_hour} and end at {end_hour} got accuracy: {accuracy:.2f}')
                print('==============================================================================')

            accs_per_shift.append(accuracy)
        final_accs.append(accs_per_shift)
        accs_per_shift = []

    return final_accs