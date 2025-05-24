import yfinance as yf
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier



def main():
    # Ask the user what company stock ticker
    ticker = input("Enter the stock symbol (e.g., AAPL, GOOG): ").upper()

    # Download the stock data
    print(f"Downloading data for {ticker}....")
    data = yf.download(ticker, start="2018-01-01", end="2023-12-31")

    # Check if the data was retrieved
    if data.empty:
        print("No data found. Please check the stock symbol.")
        return

    # Use previous days data to predict the next days closing movement
    data["Return"] = data["Close"].pct_change()
    data["Target"] = (data["Return"].shift(-1) > 0).astype(int) # 1 if next day is up, else 0
    data["Prev_Close"] = data["Close"].shift(1)
    data["Prev_Return"] = data["Return"].shift(1)

    # Drop the rows that contain NaN values
    data.dropna(inplace=True)

    # Features: (Can choose more)
    features = ["Open", "High", "Low", "Close", "Volume", "Prev_Close", "Prev_Return"]
    X = data[features]
    y = data["Target"]

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)
    acc_nb = accuracy_score(y_test, y_pred_nb)

    # Train Linear Regression (convert regression output to class)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_pred_lr_class = (y_pred_lr > 0.5).astype(int)
    acc_lr = accuracy_score(y_test, y_pred_lr_class)

    # Train MLP Neural Network
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    mlp.fit(X_train, y_train)
    y_pred_mlp = mlp.predict(X_test)
    acc_mlp = accuracy_score(y_test, y_pred_mlp)

    # Show results
    print(f"\n Model Accuracies for {ticker}:")
    print(f"Naive Bayes Accuracy:      {acc_nb:.2%}")
    print(f"Linear Regression Accuracy: {acc_lr:.2%}")
    print(f"Neural Network Accuracy:    {acc_mlp:.2%}")


main()