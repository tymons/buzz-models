import getopt
import sys
import csv
import os
import glob
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path, PurePath

from smartula_ask import SmartulaAsk
from smartula_sound import SmartulaSound
from sklearn.manifold import TSNE
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource


def print_with_bokeh(data_frame):

    data_frame['colors'] = ["#003399" if elfield == "True" else "#ff0000" for elfield in data_frame['elfield']]
    source = ColumnDataSource(data=data_frame)

    tools = "hover,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select," \
            "poly_select,lasso_select, "
    tooltips = [
        ("timestamp", "@timestamp"),
        ("class", "@elfield")
    ]
    p = figure(tools=tools, tooltips=tooltips)
    p.scatter(x='x', y='y', fill_color='colors', fill_alpha=0.4, source=source, size=15, line_color=None)
    output_file("color_scatter.html", title="color_scatter.py example")
    show(p)  # open a browser


def __prepare_mfcc_in_folder(folder_with_sounds, mfcc_folder_name):
    """
    Function for reading sounds from specific folder and extracting stationary mfcc values
    :type folder_with_sounds: str   Folder name where sound should be placed
    :type mfcc_folder_name: str     Folder where mfccs will be saved
    """
    list_of_audios = __change_directory_prepare_sound_with_mfcc(folder_with_sounds,
                                                                [("2019-06-04T18-22-00", "2019-06-04T20-30-00"),
                                                                 ("2019-06-05T20-46-00", "2019-06-05T23-48-00"),
                                                                 ("2019-06-06T22-23-00", "2019-06-07T05-52-00")])
    try:
        os.mkdir(mfcc_folder_name)
    except FileExistsError:
        print("Folder " + mfcc_folder_name + " already exists.")
    for smartula_audio in list_of_audios:
        save_to_file(mfcc_folder_name + "/" +
                     str(smartula_audio.electromagnetic_field_on) + " " + smartula_audio.timestamp,
                     smartula_audio.mfcc_feature_vector)


def save_to_file(filename, samples):
    """
    Function for saving samples in csv file
    :param filename:    filename
    :param samples:     list of samples
    """
    with open(filename, mode='w') as csv_file:
        file_writer = csv.writer(csv_file, delimiter=' ', lineterminator='\n')
        for sample in samples:
            file_writer.writerow([sample])


def read_from_csv(file_name):
    """
    Function for reading samples from csv file
    :param file_name:   filename
    :return:            sample list
    """
    samples = []
    with open(file_name, mode='r') as csv_file:
        file_reader = csv.reader(csv_file, delimiter=' ', lineterminator='\n')
        for row in file_reader:
            if row:
                samples.append(float(row[0]))

    return samples


def __parse_range_string(sound_ids):
    for char in ["[", "]"]:
        sound_ids = sound_ids.replace(char, '')

    sound_from_id, sound_to_id = sound_ids.split(':')
    sound_from_id, sound_to_id = int(sound_from_id), int(sound_to_id)
    return list(range(sound_from_id, sound_to_id + 1))


def __get_sound_and_save_to_file(sma, sound_id):
    samples, timestamp = sma.get_sound(1300001, sound_id)
    if os.name == 'nt':
        timestamp = timestamp.replace(":", "-")

    save_to_file("csv/" + str(timestamp) + ".csv", samples)
    print('Success at sound (' + str(sound_id) + ') download!')


def __affected(datetime_string, list_of_tuples_interval):
    is_affected = False
    date_time_obj = datetime.datetime.strptime(datetime_string, "%Y-%m-%dT%H-%M-%S")
    for interval in list_of_tuples_interval:
        from_datetime_obj = datetime.datetime.strptime(interval[0], "%Y-%m-%dT%H-%M-%S")
        to_datetime_obj = datetime.datetime.strptime(interval[1], "%Y-%m-%dT%H-%M-%S")
        is_affected |= (from_datetime_obj < date_time_obj < to_datetime_obj)

    return is_affected


def __change_directory_prepare_sound_with_mfcc(folder_name, list_of_tuples_interval):
    os.chdir(folder_name)
    all_filenames = [i for i in glob.glob("*.{}".format("csv"))]
    list_of_audios = [SmartulaSound(f, __affected(f.replace(".csv", ""), list_of_tuples_interval),
                                    np.ravel(pd.read_csv(f, header=None)))
                      for f in all_filenames]
    return list_of_audios


def __read_mfcc_from_folder(folder_name):
    data_folder = Path(folder_name)
    files_to_open = data_folder / "*.csv"

    all_filenames = [i for i in glob.glob(str(files_to_open))]
    # all_filenames = all_filenames[:10]
    list_of_ss_mfcc = [SmartulaSound(PurePath(f).name.split(" ")[1].replace(".csv", ""),
                                     PurePath(f).name.split(" ")[0],
                                     samples=None, mfcc=np.ravel(pd.read_csv(f, header=None)))
                       for f in all_filenames]
    return list_of_ss_mfcc


def main(argv):
    username = ''
    password = ''
    file_name = ''
    sound = 0

    try:
        opts, args = getopt.getopt(argv, "hu:p:s:f:", ["username=", "password=", "sound=", "file="])
    except getopt.GetoptError:
        print('smartula_analyze.py [-u <username> -p <password> -s <sound_id>] [--f <path_to_file>]')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('smartula_analyze.py -u <username> -p <password>')
            sys.exit()
        elif opt in ("-u", "--username"):
            username = arg
        elif opt in ("-s", "--sound"):
            sound = arg
        elif opt in ("-p", "--password"):
            password = arg
        elif opt in ("-f", "--file"):
            file_name = arg

    if username and password:
        sma = SmartulaAsk(username, password, "http://cejrowskidev.com:8884/")

        # Create folder for CSVs
        try:
            os.mkdir("csv")
        except FileExistsError:
            print("Folder already exists.")

        # Get actual samples
        try:
            sound_id = int(sound)
            __get_sound_and_save_to_file(sma, sound_id)
        except ValueError:
            sound_ids = __parse_range_string(sound)
            for sound_id in sound_ids:
                __get_sound_and_save_to_file(sma, sound_id)
    else:
        # Analyze whole csv folder
        print("Smartula analyze start!")
        # __prepare_mfcc_in_folder("csv/", "mfcc-electromagnetic-field")
        list_of_smartula_mfcc = __read_mfcc_from_folder("csv/mfcc-electromagnetic-field/")

        mfccs_embedded = TSNE(n_components=2, perplexity=5, learning_rate=500, n_iter=5000, verbose=1) \
            .fit_transform([ss.mfcc_feature_vector for ss in list_of_smartula_mfcc])

        df_subset = pd.DataFrame()
        df_subset['x'] = mfccs_embedded[:, 0]
        df_subset['y'] = mfccs_embedded[:, 1]
        df_subset['elfield'] = [ss.electromagnetic_field_on for ss in list_of_smartula_mfcc]
        df_subset['timestamp'] = [ss.timestamp for ss in list_of_smartula_mfcc]

        print_with_bokeh(df_subset)

        print("Smartula analyze end!")


if __name__ == "__main__":
    main(sys.argv[1:])
