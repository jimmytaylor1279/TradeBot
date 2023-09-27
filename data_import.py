"""
This file interacts with the Alpaca-py API to fetch the necessary data,
such as the historical data for training your neural network and the
real-time data for making trades.
"""
import asyncio
from typing import Callable
from alpaca.data.live.stock import StockDataStream
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

def get_historical_data(symbol_or_symbols, timeframe, start_date, end_date):
    """
    Fetches historical data for the specified symbols and timeframe.

    Parameters:
    symbol_or_symbols (str or list[str]): The symbol or list of symbols to fetch
    historical data for. timeframe (str): The timeframe for the historical data. This should
    be one of the TimeFrame values. start_date (str): The start date for the historical data
    in the format YYYY-MM-DD. end_date (str): The end date for the historical data in the
    format YYYY-MM-DD.

    Returns:
    pandas.DataFrame: A DataFrame containing the historical data.

    Raises:
    ValueError: If the timeframe is not a valid TimeFrame value.
    """
    valid_timeframes = ['ONE_MIN', 'FIVE_MIN', 'FIFTEEN_MIN', 'ONE_HOUR', 'ONE_DAY']

    # Check if the timeframe is valid
    if timeframe not in valid_timeframes:
        raise ValueError(f"Invalid timeframe: {timeframe}. Must be one of: {valid_timeframes}")

    # Instantiate the crypto historical data client
    client = CryptoHistoricalDataClient()

    # Create the request parameters
    request_params = CryptoBarsRequest(
        symbol_or_symbols=symbol_or_symbols,
        timeframe=getattr(TimeFrame, timeframe),
        start=start_date,
        end=end_date
    )

    try:
        # Retrieve the data
        data = client.get_crypto_bars(request_params)

        # Convert to a DataFrame
        return data.df
    except Exception as error:
        print(f"An error occurred while fetching historical data: {error}")
        return None

async def get_live_stream_data(symbol: str, data_type: str, handler: Callable):
    """
    Fetches live stream data for the specified symbol.

    Parameters:
    symbol (str): The symbol to fetch live data for.
    data_type (str): The type of live data to fetch. This should be one of the following: 
                     'bars', 'daily_bars', 'quotes', 'trades', 'updated_bars'.
    handler (Callable): The coroutine callback function to handle live data.

    Returns:
    None

    Raises:
    ValueError: If the data_type is not a valid value.
    """
    valid_data_types = ['bars', 'daily_bars', 'quotes', 'trades', 'updated_bars']

    # Check if the data_type is valid
    if data_type not in valid_data_types:
        raise ValueError(f"Invalid data_type: {data_type}. Must be one of: {valid_data_types}")

    # Instantiate the stock data stream client
    client = StockDataStream('your_api_key', 'your_secret_key')

    try:
        # Start the websocket connection's event loop
        asyncio.run(client.run())

        # Subscribe to the specified live data type
        getattr(client, f'subscribe_{data_type}')(handler, symbol)

    except Exception as error:
        print(f"An error occurred while fetching live stream data: {error}")
        return None
