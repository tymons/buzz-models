import getopt
import sys
import csv
from smartula_ask import SmartulaAsk
from smartula_sound import SmartulaSound


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


def main(argv):
    username = ''
    password = ''
    file_name = ''
    sound_id = 0

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
            sound_id = arg
        elif opt in ("-p", "--password"):
            password = arg
        elif opt in ("-f", "--file"):
            file_name = arg

    if file_name:
        # Read CSV
        samples = read_from_csv(file_name)
        sms = SmartulaSound(samples)
        sms.get_fft()
        print('End!')

    else:
        sma = SmartulaAsk(username, password, "http://cejrowskidev.com:8884/")
        samples = sma.get_sound(1300001, sound_id)
        save_to_file((str(sound_id) + '.csv'), samples)
        print('Success at sound (' + str(sound_id) + ') download!')


if __name__ == "__main__":
    main(sys.argv[1:])
