from keras.layers import Dense, Dropout, Activation, Conv1D, Conv2D, Flatten, MaxPooling1D, MaxPooling2D, LeakyReLU, AveragePooling1D
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from numpy import loadtxt


def lstm_model(units):

    model = Sequential()
    model.add(LSTM(units[1], input_shape=(units[0], 1), return_sequences=True))
    model.add(LSTM(units[2]))
    model.add(Dropout(0.1))
    model.add(Dense(units[3], activation='relu'))

    return model


def cnn_model():

    model = Sequential()

    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(Conv1D(filters=16, kernel_size=3, activation='relu'))
    model.add(Conv1D(filters=8, kernel_size=3, activation='relu'))
    model.add(Conv1D(filters=4, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(1, activation='relu'))

    return model