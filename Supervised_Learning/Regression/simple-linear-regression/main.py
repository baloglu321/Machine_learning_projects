import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from utils import calc_hist, update_model, predict_next_day_price

def update():
    history = calc_hist()
    model = update_model()
    predict = predict_next_day_price(history[-1][1], model)
    history = pd.DataFrame(history, columns=["Date", "Real", "Predict"])
    return history, model, predict


def start_streamlit():
    history, model, predict = update()
    st.header("Simple Linear Regression", divider="blue")
    st.write(
        "This page contains an algorithm that predicts the next day's close of nvidia stock based on the previous day's movements using a Simple Linear Regression model"
    )
    st.divider()
    with st.spinner("Wait for it..."):
        number = st.number_input("Insert a close value")
        predict_close_value = predict_next_day_price(number, model)
        st.write(
            f"If it closes at {number} today, it will close at an estimated value of {round(predict_close_value,2)} tomorrow. "
        )
    st.divider()
    with st.spinner("Wait for it..."):
        days = st.slider("How many days last?", 0, 250, int(predict))
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
    st.caption('Improving by MEB')

start_streamlit()
