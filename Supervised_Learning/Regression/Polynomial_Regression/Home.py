import streamlit as st
import pandas as pd
from utils import pre_process_data, predict_price, update_model
import pickle

model = update_model()


def start_streamlit():
    df, data_num, raw_data = pre_process_data()

    string_columns = raw_data.select_dtypes(include="object")
    numeric_columns = raw_data.select_dtypes(include="number")
    string_unq = {
        col: sorted(string_columns[col].unique()) for col in string_columns.columns
    }
    min_values = numeric_columns.min()
    max_values = numeric_columns.max()
    st.header("The features by the predicted car", divider="blue")
    col1, col2, col3 = st.columns(3)
    with col1:
        brand = st.selectbox("Car Brand", tuple(string_unq["Manufacturer"]))
        leather = st.selectbox(
            "Leather interior", tuple(string_unq["Leather interior"])
        )
        drive_whells = st.selectbox("Drive wheels", tuple(string_unq["Drive wheels"]))
    with col2:
        car_model = st.selectbox("Car Model", tuple(string_unq["Model"]))
        fuel_type = st.selectbox("Fuel Type", tuple(string_unq["Fuel type"]))
        whell = st.selectbox("Wheel", tuple(string_unq["Wheel"]))
    with col3:
        category = st.selectbox("Car Category", tuple(string_unq["Category"]))
        gear_type = st.selectbox("Gear box type", tuple(string_unq["Gear box type"]))
        color = st.selectbox("Color", tuple(string_unq["Color"]))
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        engine_volume = st.slider(
            "Engine Volume", min_values["Engine volume"], max_values["Engine volume"]
        )
        cylinders = st.slider(
            "Cylinders", int(min_values["Cylinders"]), int(max_values["Cylinders"])
        )
    with col2:
        airbags = st.slider(
            "Airbags", int(min_values["Airbags"]), int(max_values["Airbags"])
        )
    with col3:
        age = st.slider("Age", int(min_values["Age"]), int(max_values["Age"]))
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        levy = st.number_input("Levy", min_values["Levy"], max_values["Levy"])
    with col2:
        mileage = st.number_input(
            "Mileage", min_values["Mileage"], max_values["Mileage"]
        )

    pred_var = [
        [
            brand,
            car_model,
            category,
            leather,
            fuel_type,
            gear_type,
            drive_whells,
            whell,
            color,
            levy,
            engine_volume,
            mileage,
            cylinders,
            airbags,
            age,
        ]
    ]
    columns = [
        "Manufacturer",
        "Model",
        "Category",
        "Leather interior",
        "Fuel type",
        "Gear box type",
        "Drive wheels",
        "Wheel",
        "Color",
        "Levy",
        "Engine volume",
        "Mileage",
        "Cylinders",
        "Airbags",
        "Age",
    ]
    pred_x = pd.DataFrame(pred_var, columns=columns)
    predict = predict_price(pred_x)
    st.write(predict)


start_streamlit()
