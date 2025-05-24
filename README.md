# Stock Price Direction Predictor

This project predicts the next day's stock price movement (up or down) using historical stock data and three different machine learning models:

- **Naive Bayes**
- **Linear Regression**
- **Neural Network (MLPClassifier)**

---

## Features Used

- Open price
- High price
- Low price
- Close price
- Trading volume

---

## How It Works

1. The user inputs a stock ticker symbol (e.g., AAPL, GOOG, NVDA).
2. The script downloads historical daily stock data from Yahoo Finance (from 2018 to 2023).
3. It calculates daily returns and sets the target variable based on whether the next day’s return is positive (1) or negative (0).
4. The data is preprocessed and split into training and testing sets.
5. Three models are trained and tested to predict stock movement direction.
6. The script outputs the accuracy for each model.

---
