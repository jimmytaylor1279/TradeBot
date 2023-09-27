"""
This program is a trading bot that uses the Alpaca API to trade stocks.
"""
import tkinter as tk
from neural_network import NeuralNetwork
from trading import Trader
from symbol_utilities import SymbolUtilities
from data_import import DataImporter
from gui import GUI

class Main:
    """
    The Main class is the central class that connects and
    manages all the components of the trading bot.
    """

    def __init__(self):
        """
        Initialize all the components of the trading bot.
        """
        self.symbol_utilities = SymbolUtilities()
        self.data_importer = DataImporter(self.symbol_utilities)
        self.neural_network = NeuralNetwork(self.data_importer)
        self.trader = Trader(self.neural_network, self.data_importer)
        self.gui = GUI(self.trader)

    def run(self):
        """
        Run the main process of the trading bot. This includes
        fetching the data, training the neural network,
        starting the trading process, and launching the GUI.
        """
        historical_data, real_time_data = self.data_importer.fetch_data()
        self.neural_network.train(historical_data)
        self.trader.start_trading(real_time_data)
        self.gui.mainloop()


if __name__ == "__main__":
    app = Main()
    app.run()
