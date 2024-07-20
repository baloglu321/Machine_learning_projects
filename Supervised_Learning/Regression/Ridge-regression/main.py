import pandas as pd
import streamlit as st
from datetime import datetime
from utils import calc_hist, update_model, predict_next_day_price, pre_process_data


def update():
    history = calc_hist()
    model, mse, r2, ridge_coefficients = update_model()
    history = pd.DataFrame(history, columns=["Date", "Real", "Predict"])
    return history, model, mse, r2, ridge_coefficients


def start_streamlit():
    history, model, mse, r2, ridge_coefficients = update()
    st.header("Ridge Regression", divider="blue")
    st.write(
        "This page contains an algorithm that predicts the next day's close of nvidia stock based on the previous day's movements using a Ridge Regression model"
    )
    st.divider()
    st.header(":blue[Model Status]", divider="blue")
    st.table(ridge_coefficients)
    st.caption(
        "INFO:l In the table above you can see how effectively the ridge model keeps which of the data."
    )
    st.caption(f":blue[MSE:{round(mse,2)}] and :blue[R2:{round(r2,5)}]")
    st.divider()

    with st.spinner("Wait for it..."):
        today = datetime.today()
        start_date = datetime(2020, 1, 1).date()
        data = pre_process_data(start_date, today=today)
        current_low = data.iloc[-1]["Low"]
        current_high = data.iloc[-1]["High"]
        current_open = data.iloc[-1]["Open"]
        current_close = data.iloc[-1]["Close"]

        low = st.number_input(
            "Insert a lower value", min_value=1.0, value=round(current_low, 2)
        )
        high = st.number_input(
            "Insert a higher value", min_value=1.0, value=round(current_high, 2)
        )
        open = st.number_input(
            "Insert a open value", min_value=1.0, value=round(current_open, 2)
        )
        close = st.number_input(
            "Insert a close value", min_value=1.0, value=round(current_close, 2)
        )

        predict_close_value = predict_next_day_price(low, high, open, close, model)
        gap = close - predict_close_value
        if gap <= 0:
            text = f"If it closes at :blue[{round(close,2)}] today, it will close at an estimated value of  :green[{round(predict_close_value,2)}] tomorrow."
        elif gap > 0:
            text = f"If it closes at :blue[{round(close,2)}] today, it will close at an estimated value of  :red[{round(predict_close_value,2)}] tomorrow."
        st.caption(text)
    with st.spinner("Wait for it..."):
        days = st.slider("How many days last?", 0, 250, 150)
        option = st.selectbox(
            "Which graphs should be drawn?",
            ("Real & Predict", "Just Real", "Just Predict"),
        )
        if option == "Real & Predict":
            conf = ["Real", "Predict"]
            color = ["#FF0000", "#0000FF"]
        elif option == "Just Real":
            conf = ["Real"]
            color = ["#FF0000"]
        elif option == "Just Predict":
            conf = ["Predict"]
            color = ["#0000FF"]
        st.line_chart(history[-days:], x="Date", y=conf, color=color)

    st.caption(
        "INFO:l Ridge Regression is a variant of linear regression and uses regularisation to increase its generalisation ability. Ridge Regression works by adding a penalisation term known as the L2 norm. This penalisation involves the sum of the squares of the regression coefficients and thus attempts to shrink large coefficients. Ridge Regression is used to prevent overfitting, but unlike Lasso Regression, it does not reduce the coefficients to zero."
    )
    st.divider()
    st.caption(":blue[Improving by MEB]")
    st.divider()


start_streamlit()
