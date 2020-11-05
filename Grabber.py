import requests
from matplotlib import pyplot as plt
from datetime import datetime
from pathlib import Path
import os
import sys
from time import sleep
import csv


# Tells api to grab demo data set
api_key = 'demo'
API_key = 'OJIP3R50HCID92MX'

Username = os.getlogin()
Stock_Prices = Path('C:/Users/' + Username + '/Documents/StockTrader/Files')
Charts_Directory = Path('C:/Users/' + Username + '/Documents/StockTrader/Charts')
Symbols_Directory = Path('C:/Users/' + Username + '/Documents/StockTrader/')

folders = []
field_names = ['1. open', '2. high', '3. low', '4. close', '5. volume']

clean_names = ['open', 'high', 'low', 'close', 'volume']

folders.append(Stock_Prices)
folders.append(Charts_Directory)


def get_symbols():
    file_path = Symbols_Directory / 'symbols.txt'
    fd = open(file_path, 'r')
    symbols = fd.readlines()

    for each in range(len(symbols)):
        print(each)
        symbols[each] = symbols[each].replace('\n', '')
    return symbols


def check_folder(name):
    if not os.path.exists(name):
        os.makedirs(name)

def main():

    # Paths to be sent to preceding scripts
    paths = []

    symbols = get_symbols()
    for each in folders:
        check_folder(each)
    for symbol in symbols:
        path = Stock_Prices / symbol
        check_folder(path)
        #  API only allows 5 Pulls per minute
        grab_history(symbol, path)
        # sleep(60)
        grab_intraday(symbol, path)
        print(symbol + ' Done')
        paths.append(path)
        # sleep(60)
    # sleep(15)
    print('Done')
    #sys.exit()
    sys.stdout.flush()
    return paths

def grab_intraday(symbol, path):
    data = None
    r = requests.get('https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&interval=15min&outputsize=full&symbol=' + symbol + '&apikey=' + API_key)
    if r.status_code == 200:
        print('Successfully Connected')
        data = r.json()
        # print(data)
        file = symbol + 'intra.csv'
        file = path / file
        fd = open(file, 'w+')
        fd.write('DATE,OPEN,HIGH,LOW,CLOSE,VOLUME\n')
        for key in data:
            if key == 'Time Series (15min)':
                # print(data[key])
                for attribute in reversed(list(data[key])):
                    fd.write(attribute + ',')
                    for each_field in range(len(field_names)):
                        if each_field == len(field_names) - 1:
                            fd.write(data[key][attribute][field_names[each_field]])
                        else:
                            fd.write(data[key][attribute][field_names[each_field]] + ',')
                    fd.write('\n')
        fd.close()
                #fd.write(data[key] + '\n')


def grab_history(symbol, path):
    data = None
    r = requests.get(
        'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&outputsize=full&symbol=' + symbol + '&apikey=' + API_key)
    if (r.status_code == 200):
        print('Successfully Connected')
        # print(r.content)
        data = r.json()
        file = path / (symbol + '.csv')
        fd = open(file, 'w+')
        fd.write('DATE,OPEN,HIGH,LOW,CLOSE,VOLUME\n')
    else:
        print('Failed Connection, terminating application')
        fd = None
        sys.exit()


    x = []
    y = []
    for each in range(len(field_names)):
        y.append([])

    for key in data:
        if key == 'Time Series (Daily)':
            for attribute in reversed(list(data[key])):
                fd.write(attribute + ',')
                for each_field in range(len(field_names)):
                    fd.write(data[key][attribute][field_names[each_field]])
                    y[each_field].append(data[key][attribute][field_names[each_field]])
                    if each_field != 4:
                        fd.write(',')
                fd.write('\n')

                x.append(attribute)

    for each_day in range(len(x)):
        date_obj = datetime.strptime(x[each_day], '%Y-%m-%d')
        x[each_day] = date_obj
    #fd.flush()
    fd.close()
    # for each_day in range(len(y)):
        #y[each_day] = float(y[each_day])

    for each_trace in range(len(y)):
        for each_set in range(len(y[each_trace])):
            y[each_trace][each_set] = float(y[each_trace][each_set])
        plot_data(x, y[each_trace], (symbol + clean_names[each_trace]))
    # exit()
    # print()


def plot_data(x_list, y_list, symbol):
    plt.figure(figsize=(16, 8))
    plt.plot(x_list, y_list)
    # plt.show(block=False)
    plt.savefig(Charts_Directory / symbol)
    #plt.pause(3)
    #plt.close()
    return


if __name__ == '__main__':
    main()
