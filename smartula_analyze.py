import getopt
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from smartula_ask import SmartulaAsk


def main(argv):
    username = ''
    password = ''
    file_name = ''

    try:
        opts, args = getopt.getopt(argv, "hu:p:f:", ["username=", "password=", "file="])
    except getopt.GetoptError:
        print('smartula_analyze.py [-u <username> -p <password>] [--f <path_to_file>]')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('smartula_analyze.py -u <username> -p <password>')
            sys.exit()
        elif opt in ("-u", "--username"):
            username = arg
        elif opt in ("-p", "--password"):
            password = arg
        elif opt in ("-f", "--file"):
            file_name = arg

    if file_name:
        # Read CSV
        samples = pd.read_csv(file_name)
        print(samples.head())
        samples.plot()
        plt.show()

    else:
        sma = SmartulaAsk(username, password, "http://cejrowskidev.com:8884/")
        samples = sma.get_sound(1300001, 119)
        print(samples)


if __name__ == "__main__":
    main(sys.argv[1:])
