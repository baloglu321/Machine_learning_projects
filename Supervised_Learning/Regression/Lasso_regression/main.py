import pandas as pd
import streamlit as st
from datetime import datetime
from utils import calc_hist, update_model, predict_next_day_price, pre_process_data


def update():
    history = calc_hist()
    model, mse, r2,lasso_coefficients = update_model()
    history = pd.DataFrame(history, columns=["Date", "Real", "Predict"])
    return history, model, mse, r2,lasso_coefficients


def start_streamlit():
    history, model, mse, r2,lasso_coefficients = update()
    st.header("Lasso Regression", divider="blue")
    st.write(
        "This page contains an algorithm that predicts the next day's close of nvidia stock based on the previous day's movements using a Lasso Regression model"
    )
    st.divider()
    st.header(':blue[Model Status]',divider='blue')
    st.table(lasso_coefficients)
    st.caption("INFO:l In the table above you can see how effectively the lasso model keeps which of the data.")
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
        st.line_chart(history[-days:], x="Date", y=conf, color=color)  # Optional
    
    st.caption("INFO:l Lasso (Least Absolute Shrinkage and Selection Operator) Regression is a variant of linear regression and is used for feature selection. Lasso Regression increases the simplicity and generalizability of the model by making unimportant features in the data set zero. This is especially useful in data sets with a large number of features. Lasso Regression works by adding a penalization term known as the L1 norm. This penalization involves the absolute value of the sum of the regression coefficients, thus causing some coefficients to be zero.")    
    st.divider()
    st.caption("Improving by MEB")
    st.divider()

start_streamlit()
