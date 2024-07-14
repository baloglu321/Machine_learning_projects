import numpy as np
from sklearn.linear_model import LinearRegression
import yfinance as yf
from datetime import datetime, timedelta


def pre_process_data(start_date, today):
    today = today.date()
    yesterday = today - timedelta(days=-1)
    ticker = "NVDA"
    data = yf.download(ticker, start=str(start_date), end=str(yesterday))
    data["Next_Close"] = data["Close"].shift(-1)
    data = data.dropna()
    return data


def predict_next_day_price(close_price, model):
    close_price = np.array(close_price).reshape(-1, 1)
    prediction = model.predict(close_price)
    return float(prediction[0])


def first_model():
    today = datetime.today()
    start_date = datetime(2020, 1, 1).date()

    data = pre_process_data(start_date, today=today)

    X = data[["Close"]][:100].values
    y = data["Next_Close"][:100].values
    model = LinearRegression()
    model.fit(X, y)
    return model


def calc_hist():
    today = datetime.today()
    start_date = datetime(2020, 1, 1).date()
    data = pre_process_data(start_date, today=today)

    model = first_model()
    history = []
    for index in range(100, len(data)):
        date = data.index[index].date().strftime("%Y-%m-%d")
        real_value = data["Close"][index]
        predict = predict_next_day_price(data["Close"][index], model)
        history.append([date, real_value, predict])
        X = data[["Close"]][:index].values
        y = data["Next_Close"][:index].values
        model = LinearRegression()
        model.fit(X, y)
    return history


def update_model():
    today = datetime.today()
    start_date = datetime(2020, 1, 1).date()

    data = pre_process_data(start_date, today=today)

    X = data[["Close"]].values
    y = data["Next_Close"].values
    model = LinearRegression()
    model.fit(X, y)
    return model
