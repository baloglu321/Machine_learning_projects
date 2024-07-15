import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd

def pre_process_data(start_date, today):
    today = today.date()
    yesterday = today - timedelta(days=-1)
    ticker = "NVDA"
    data = yf.download(ticker, start=str(start_date), end=str(yesterday))
    data["Next_Close"] = data["Close"].shift(-1)
    data = data.dropna()
    return data


def predict_next_day_price(low,high,open_price,close_price,model):
    low = np.array([low])
    high = np.array([high])
    open_price = np.array([open_price])
    close_price = np.array([close_price])
    data = pd.DataFrame(data={'low': low, 'high': high, 'open': open_price, 'close': close_price})
    prediction = model.predict(data)
    return float(prediction[0])


def first_model():
    today = datetime.today()
    start_date = datetime(2020, 1, 1).date()

    data = pre_process_data(start_date, today=today)

    X = data[["Low","High","Open","Close"]][:100].values
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
        real_value = data["Next_Close"][index]
        predict = predict_next_day_price(data["Low"][index],data["High"][index],data["Open"][index],data["Close"][index], model)
        history.append([date, real_value, predict])
        X = data[["Low","High","Open","Close"]][:index].values
        y = data["Next_Close"][:index].values
        model = LinearRegression()
        model.fit(X, y)
    return history


def update_model():
    today = datetime.today()
    start_date = datetime(2020, 1, 1).date()

    data = pre_process_data(start_date, today=today)

    
    X = data[["Low","High","Open","Close"]].values
    y = data["Next_Close"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return model,mse,r2
