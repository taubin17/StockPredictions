import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import math
import sys
from termcolor import colored

Username = os.getlogin()

old_day = 0

Stock_Prices = Path('C:/Users/' + Username + '/Documents/StockTrader/Files')
Model_Directory = Path('C:/Users/' + Username + '/Documents/StockTrader/Models')
Symbols_Directory = Path('C:/Users/' + Username + '/Documents/StockTrader/')


days_to_look_back = 1500
days_to_predict = 30


def main(paths):

    # Load a list of all models that can run predictions (ones that have an associative LSTM)
    models = [dI for dI in os.listdir(Model_Directory) if os.path.isdir(os.path.join(Model_Directory, dI))]

    file_path = Symbols_Directory / 'symbols.txt'

    symbol_file = open(file_path, 'r')

    models_to_predict = symbol_file.readlines()

    for entry in range(len(models_to_predict)):
        models_to_predict[entry] = models_to_predict[entry].rstrip()

    for each_path in paths:
        for each_file in os.listdir(each_path):
            print(each_path, each_file)
            print(f'{each_file} Running Predictions.')
            predict(each_path, each_file)
            print(f'{each_file} Predictions Complete.')
        else:
            continue

    sys.exit()


def read_csv(file, columns):
    data = pd.read_csv(file, usecols=[columns])
    return data


def predict(item_path, item_name):
    print(item_path)
    file = item_name
    item_name = item_name.replace('.csv', '')
    data = read_csv(item_path / file, 1)
    dataset = data.values

    global old_day

    date = read_csv(item_path / file, 0)

    training_data_len = math.ceil(len(dataset) * .8)

    scalar = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scalar.fit_transform(dataset)

    training_data = data_scaled[0:training_data_len, :]

    x_train = []
    y_train = []

    time_steps = 60
    feature_count = 1

    for i in range(time_steps, len(training_data)):
        x_train.append(training_data[i - time_steps: i, 0])
        y_train.append(training_data[i, 0])

    test_data = data_scaled[training_data_len - time_steps:, :]

    x_test = []

    for i in range(time_steps, len(test_data)):
        x_test.append(test_data[i - time_steps: i, 0])

    x_test = np.array(x_test)

    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], feature_count))

    lstm = tf.keras.models.load_model(Model_Directory / item_name)

    answer_predicted = lstm.predict(x_test)

    for predictions in range(days_to_predict):
        x_test, answer_predicted = forecasting(lstm, answer_predicted, x_test, scalar)

    answer_predicted = scalar.inverse_transform(answer_predicted)

    train = data[:training_data_len]
    valid = data[training_data_len:]

    train_dates = []
    predict_dates = []

    dates_list = date.values.tolist()

    for items in range(len(train['OPEN'])):
        train_dates.append(dates_list[items][-1])

    train_dates = dates.datestr2num(train_dates)

    prediction_index = len(train['OPEN'])

    for items in range(prediction_index, (prediction_index + len(answer_predicted) - days_to_predict)):
        predict_dates.append(dates_list[items][-1])

    predict_dates = dates.datestr2num(predict_dates)

    forecasted_predict_dates = list(predict_dates)

    date_diff = predict_dates[-1] - predict_dates[-2]

    for forecasted_dates in range((prediction_index + len(answer_predicted) - days_to_predict),
                                  (prediction_index + len(answer_predicted))):
        new_date = date_diff + forecasted_predict_dates[-1]
        forecasted_predict_dates.append(new_date)

    forecasted_predict_dates = np.array(forecasted_predict_dates)

    old_day = 0

    plt.figure(figsize=(16, 8))
    plt.title(item_name + ' Predicted Prices')
    plt.xlabel('Day of data')
    plt.ylabel('Price of ' + item_name)
    plt.plot_date(forecasted_predict_dates, answer_predicted, 'r', xdate=True)
    plt.plot_date(predict_dates, valid, 'y', xdate=True)
    plt.plot_date(train_dates, train['OPEN'], 'g', xdate=True)
    plt.legend(['Stock Price Predicted', 'Stock Price Absolute', 'Stock Price Trained'])

    plot_file = item_name + ' month away.png'

    plt.savefig(Model_Directory / item_name / plot_file)

    plt.close()

    valid_list = list(valid['OPEN'])

    plot_red_vs_green(predict_dates, valid_list, 'UpsAndDowns.png', item_name)
    plot_red_vs_green(forecasted_predict_dates, answer_predicted, 'ForecastedUpsAndDowns', item_name)

    only_forecasted_x = []
    only_forecasted_y = []

    for points in range(len(forecasted_predict_dates) - days_to_predict, len(forecasted_predict_dates)):
        only_forecasted_x.append(forecasted_predict_dates[points])
        only_forecasted_y.append(answer_predicted[points])

    plot_red_vs_green(only_forecasted_x, only_forecasted_y, 'OnlyForecasted', item_name)

    plt.figure(figsize=(16, 8))
    plt.title(item_name + 'Test')
    plt.xlabel('Day of data')
    plt.ylabel('Price of ' + item_name)
    plt.plot_date(forecasted_predict_dates, answer_predicted, 'b')
    plt.plot_date(predict_dates, valid, 'y')

    plot_file = item_name + 'Test2'

    plt.savefig(Model_Directory / item_name / plot_file)
    plt.close()

    return


def plot_red_vs_green(x_list, y_list, name_of_file, item_name):

    # Test figure
    plt.figure(figsize=(24, 12))
    plt.title('Test Coloring')
    plt.xlabel('Date')
    plt.ylabel('Price of ' + item_name)

    green_x = []
    green_y = []
    red_x = []
    red_y = []

    for indexer in range(len(x_list)):
        if indexer >= 2:
            if y_list[indexer] > y_list[indexer-1]:
                green_x.append(x_list[indexer-1])
                green_x.append(x_list[indexer])

                green_y.append(y_list[indexer-1])
                green_y.append(y_list[indexer])

                plt.plot_date(green_x, green_y, 'g')

                green_x = []
                green_y = []
                # green_x.append(predict_dates[indexer])
                # green_y.append(valid_list[indexer])
            else:
                # plt.plot_date(predict_dates[indexer], valid_list[indexer], 'r')
                red_x.append(x_list[indexer-1])
                red_x.append(x_list[indexer])

                red_y.append(y_list[indexer-1])
                red_y.append(y_list[indexer])

                plt.plot_date(red_x, red_y, 'r')

                red_x = []
                red_y = []

    plot_file = item_name + name_of_file

    plt.savefig(Model_Directory / item_name / plot_file)

    plt.close()


def forecasting(model, prediction_results, initial_tests, scalar):

    global old_day

    empty_new_day = np.delete(initial_tests[-1], 0)

    # Add the newest prediction to the end of the new days dataset
    new_day = np.insert(empty_new_day, -1, prediction_results[-1])

    new_day_reshaped = np.reshape(new_day, (new_day.shape[0], 1))

    list_initial_tests = initial_tests.tolist()
    list_new_day = new_day_reshaped.tolist()

    list_initial_tests.append(list_new_day)

    numpy_list = np.array(list_initial_tests)
    new_day_price = model.predict(numpy_list)
    new_day_price_fixed = scalar.inverse_transform(new_day_price)

    # print(new_day_price_fixed[-1][-1])

    if old_day != 0:
        if new_day_price_fixed[-1][-1] > old_day:
            print(colored(new_day_price_fixed[-1][-1], 'green'))
        else:
            print(colored(new_day_price_fixed[-1][-1], 'red'))
    else:
        print(new_day_price_fixed[-1][-1])

    old_day = new_day_price_fixed[-1][-1]

    return numpy_list, new_day_price


if __name__ == '__main__':
    main()
