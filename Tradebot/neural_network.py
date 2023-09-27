"""
This file contains the Keras code for creating the neural network.
"""
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler

class NeuralNetwork:
    """
    This class represents a neural network model for predicting stock prices.
    It uses Keras to define and train the model, and scikit-learn for data preprocessing.
    """
    def __init__(self, data_importer):
        """
        Initialize the neural network model.

        Args:
            data_importer (DataImporter): The object that fetches the stock data.
        """
        self.data_importer = data_importer
        self.model = self._create_model()
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def _create_model(self):
        """
        Create and compile the Keras model.

        Returns:
            The compiled Keras model.
        """
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(60, 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error')

        return model

    def train(self, symbol):
        """
        Train the model on historical data for the given symbol.

        Args:
            symbol (str): The stock symbol to train on.
        """
        historical_data = self.data_importer.fetch_historical_data(symbol)
        prices = historical_data['Close'].values.reshape(-1, 1)
        prices = self.scaler.fit_transform(prices)

        x_train = []
        y_train = []

        for i in range(60, len(prices)):
            x_train.append(prices[i-60:i, 0])
            y_train.append(prices[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        self.model.fit(x_train, y_train, epochs=50, batch_size=32)
