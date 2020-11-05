import pandas as pd
from pathlib import Path
import sys
import os

file_directory = Path('C:/Users/laura/Documents/StockTrader/Files')

data_points = 1500
symbol = 'CVV'


def read_csv(file):
    data = pd.read_csv(file)
    return data


def main(paths):

    for each_path in paths:
        for all_files in os.listdir(each_path):
            resize_file(1500, all_files, each_path)

    sys.stdout.flush()
    # sys.exit()


def resize_file(rows, file, path):


    data = read_csv(path / file)
    heading = list(data.columns)
    data_values = data.values

    heading = str(heading)
    heading = heading.replace(' ', '')

    heading = heading.replace('[', '')
    heading = heading.replace(']', '')
    heading = heading.replace("'", '')

    adjusted_data = data_values[-rows::]

    if file.find('adjusted') != -1:
        return
    else:
        file = file.replace('.csv', '_adjusted.csv')

    file_path = path / file

    fd = open(file_path, 'w+')

    fd.write(str(heading) + '\n')

    for each_row in adjusted_data:
        for each_param in range(len(each_row)):
            fd.write(str(each_row[each_param]))
            if each_param != (len(each_row) - 1):
                fd.write(',')
        fd.write('\n')
    fd.close()


if __name__ == '__main__':
    main()
