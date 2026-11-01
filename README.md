# Stock Price Predictor

A deep learning project that predicts the future closing price of a stock using a Long Short-Term Memory (LSTM) neural network. The project loads historical stock data from Yahoo Finance, scales it, trains an LSTM model on past price sequences, and predicts the next day's closing value.

## Project Overview

This project is implemented in `main.py` and focuses on forecasting the stock price of `ITC.NS` using a 60-day historical window. It demonstrates:

- Data collection from Yahoo Finance
- Time series preprocessing and normalization
- LSTM model building with Keras and TensorFlow
- Training and validation on historical test data
- Predicting the next closing price

## Features

- Uses `pandas_datareader` to fetch stock data for a selected company
- Scales stock prices using `MinMaxScaler`
- Builds a sequence dataset with a 60-day lookback window
- Trains an LSTM model with dropout layers to reduce overfitting
- Evaluates the model on later historical data
- Predicts the next day's closing price and prints the result

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- pandas_datareader
- Scikit-learn
- TensorFlow / Keras

## Project Workflow

1. Select a stock ticker and date range
2. Download historical stock data from Yahoo Finance
3. Normalize closing prices between 0 and 1
4. Create training sequences using the last 60 days of prices
5. Train the LSTM network on the sequence data
6. Test the model on a later period of stock data
7. Convert predictions back to original scale
8. Forecast the next closing value

## Setup and Installation

Make sure you have Python 3.x installed. Then install the required libraries:

```bash
pip install numpy pandas matplotlib pandas_datareader scikit-learn tensorflow
```

## Running the Project

Run the script:

```bash
python main.py
```

The model will:

- load the stock data for `ITC.NS`
- train the LSTM on the data range from 2012-01-01 to 2021-02-01
- test it on more recent data
- print the predicted next-day closing price

## Customization

You can change the stock symbol and date range in `main.py`:

```python
company = 'ITC.NS'
start = dt.datetime(2012, 1, 1)
end = dt.datetime(2021, 2, 1)
```

For example, replace `ITC.NS` with another stock ticker such as `AAPL`, `MSFT`, or `RELIANCE.NS`.

## Notes

- This project is focused on learning and experimentation with time-series forecasting.
- Stock prediction is a highly uncertain task and this model is not intended as financial advice.
- The model architecture and data window can be improved further with more tuning, feature engineering, and longer training periods.

## License

This project is for educational and learning purposes.

## Example Output

```text
prediction: [[...]]
```

The exact value depends on the model training run and market data available at the time of execution.
