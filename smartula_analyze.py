import getopt
import sys

from smartula_ask import SmartulaAsk


def main(argv):
    username = ''
    password = ''

    try:
        opts, args = getopt.getopt(argv, "hu:p:", ["username=", "password="])
    except getopt.GetoptError:
        print('smartula_analyze.py -u <username> -p <password>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('smartula_analyze.py -u <username> -p <password>')
            sys.exit()
        elif opt in ("-u", "--username"):
            username = arg
        elif opt in ("-p", "--password"):
            password = arg

    sma = SmartulaAsk(username, password, "http://cejrowskidev.com:8884/")
    samples = sma.get_sound(1300001, 119)
    print(samples)


if __name__ == "__main__":
    main(sys.argv[1:])
