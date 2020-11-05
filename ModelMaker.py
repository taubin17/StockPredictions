import math
import os
from tensorflow.keras import backend
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import sys


Username = os.getlogin()

Stock_Prices = Path('C:/Users/' + Username + '/Documents/StockTrader/Files')
Model_Directory = Path('C:/Users/' + Username + '/Documents/StockTrader/Models')


def read_csv_file(file):
    data = pd.read_csv(file, usecols=[1])
    return data


def lstm_model(x_train, y_train):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=2, batch_size=1)

    return model


def create_model(data, item_name):
    dataset = data.values

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

    x_train, y_train = np.array(x_train), np.array(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], feature_count))
    test_data = data_scaled[training_data_len - time_steps:, :]

    x_test = []

    for i in range(time_steps, len(test_data)):
        x_test.append(test_data[i - time_steps: i, 0])

    lstm = lstm_model(x_train, y_train)

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], feature_count))

    answer_predicted = lstm.predict(x_test)
    answer_predicted = scalar.inverse_transform(answer_predicted)

    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = answer_predicted

    lstm.save(Model_Directory / item_name)

    plt.figure(figsize=(16, 8))
    plt.title(item_name + ' ' + 'Price Predictions')
    plt.xlabel('Date')
    plt.ylabel('Price on Market')
    plt.plot(train['OPEN'])
    plt.plot(valid['OPEN'])
    plt.plot(valid['Predictions'])
    plt.legend(['Stock Price Trained', 'Stock Price Confirmed', 'Stock Price Predicted'])

    plot_file = item_name + '.png'

    plt.savefig(Model_Directory / item_name / plot_file)


    # Rid of our model now that it has been saved, free up space
    backend.clear_session()

    return


def create_model_directory():
    if not os.path.exists(Model_Directory):
        os.makedirs(Model_Directory)
    return


def main(paths):

    for file in paths:
        print(file)

    create_model_directory()

    count = 1
    bad_items = []

    for each_path in paths:
        for each_file in os.listdir(each_path):

            data = read_csv_file(each_path / each_file)
            item_name = each_file.replace('.csv', '')

            if item_name in os.listdir(Model_Directory):

                continue

            else:

                print(f'Model not found, creating {item_name}')

                create_model(data, item_name)

    print('All models done')

    fd = open('bad_files.txt', 'w')

    for each in bad_items:
        fd.write(each + '\n')

    fd.close()

    print('Done')

    sys.stdout.flush()

    return


if __name__ == '__main__':
    main()
