# Module 4: Time Series Forecasting - Predicting Stock Prices
# Problem 4: Predicting Stock Prices using Time Series
# Dataset: Yahoo Finance stock price data
# Objective: Use time series forecasting models to predict future stock prices.

"""
Tasks:
•	Collect historical stock data.
•	Preprocess the data: Handle missing values and visualize trends.
•	Implement ARIMA, Prophet, or LSTM models to predict future prices.
"""

# Source Code (ARIMA Example):
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Load historical stock data
df = pd.read_csv('stock_prices.csv', parse_dates=['Date'], index_col='Date')

# Data Preprocessing: Handle missing values
df.fillna(method='ffill', inplace=True)

# Visualize stock prices
df['Close'].plot(figsize=(10,6))
plt.title('Stock Price Trend')
plt.show()

# ARIMA Model
model = ARIMA(df['Close'], order=(5, 1, 0))  # (p, d, q) parameters for ARIMA
model_fit = model.fit()

# Forecast the next 30 days
forecast = model_fit.forecast(steps=30)

# Plot forecasted stock prices
plt.figure(figsize=(10,6))
plt.plot(df['Close'], label='Historical Prices')
plt.plot(pd.date_range(df.index[-1], periods=30, freq='D'), forecast, label='Forecasted Prices')
plt.title('Stock Price Forecasting')
plt.legend()
plt.show()

# Evaluate model
y_test = df['Close'][-30:]
rmse = mean_squared_error(y_test, forecast, squared=False)
print(f"RMSE: {rmse}")

"""
Outcome: Students will gain practical experience in time series forecasting using ARIMA, 
and understand how to handle and forecast stock market data.
"""