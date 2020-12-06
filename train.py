import sys
import warnings
import argparse
import numpy as np
import pandas as pd
from datasets.preprocessing import process_data
from saved_model import model
from keras.models import Model
from keras.callbacks import EarlyStopping
warnings.filterwarnings("ignore")


def train_lstm(model, x_train, y_train, name):

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    hist = model.fit(
        x_train, y_train,
        batch_size=10,
        epochs=20,
        validation_split=0.05)

    model.save('saved_model/' + name + '.h5')
    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv('saved_model/' + name + ' loss.csv', encoding='utf-8', index=False)

def train_cnn(model, x_train, y_train, name):

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    hist = model.fit(
        x_train, y_train,
        batch_size=25,
        epochs=20,
        validation_split=0.05)

    model.save('saved_model/' + name + '.h5')
    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv('saved_model/' + name + ' loss.csv', encoding='utf-8', index=False)

def lstm(x_train, y_train):
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    m = model.get_lstm([12, 64, 64, 1])
    train_lstm(m, x_train, y_train, "lstm")

def cnn(x_train, y_train):
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    m = model.get_cnn()
    train_cnn(m, x_train, y_train, "cnn")

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--model",
    #     default="cnn",
    #     help="Model to train.")
    # args = parser.parse_args()

    lag = 12
    training_data = 'datasets/train.csv'
    test_data = 'datasets/test.csv'
    x_train, y_train, _, _, _ = process_data(training_data, test_data, lag)
    
    i = 0

    while i < 2:

        lstm(x_train, y_train)

        cnn(x_train, y_train)

        i = i + 1


if __name__ == '__main__':
    main()
