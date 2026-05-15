import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import datetime

st.title("Stock Price Prediction App")
# Get user input for stock ticker and date range
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL):", "AAPL")

if ticker: 
    # Get historical stock data
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=365 * 2)

    data = yf.download(ticker, start=start_date, end=end_date)

    if not data.empty:
        st.subheader(f"Historical Data for {ticker}")
        st.line_chart(data['Close'])

        # Prepare data for prediction
        data['Date'] = data.index
        data['Date_Ordinal'] = pd.to_datetime(data['Date']).map(datetime.datetime.toordinal)

        X = data[['Date_Ordinal']]
        y = data['Close']

        # Train/Test Split
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        #Train the model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict future stock price
        y_pred = model.predict(X_test)

        st.subheader("Predicted vs Actual Stock Prices")
        fig, ax = plt.subplots()
        ax.plot(data['Date'][train_size:], y_test, label='Actual')
        ax.plot(data['Date'][train_size:], y_pred, label='Predicted')
        ax.set_xlabel('Date')
        ax.set_ylabel('Close Price')
        ax.legend()
        st.pyplot(fig)

    else:
        st.error(f"No data found for ticker {ticker}. Please enter a valid symbol.")
 
                