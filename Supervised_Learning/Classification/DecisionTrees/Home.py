import streamlit as st
import pandas as pd
from utils import pre_process_data, predict, update_model

model = update_model()


def start_streamlit():
    df = pre_process_data()

    string_columns = df.select_dtypes(include="object")
    numeric_columns = df.select_dtypes(include="number")
    string_unq = {
        col: sorted(string_columns[col].unique()) for col in string_columns.columns
    }
    min_values = numeric_columns.min()
    max_values = numeric_columns.max()
    st.header("Customer Segment Classification with Decision Tree", divider="blue")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Select Gender", tuple(string_unq["Gender"]))
        married = st.selectbox("Maried ?", tuple(string_unq["Ever_Married"]))
        graduated = st.selectbox("Graduated ?", tuple(string_unq["Graduated"]))
    with col2:
        profession = st.selectbox("Profession", tuple(string_unq["Profession"]))
        spending = st.selectbox("Spending Score", tuple(string_unq["Spending_Score"]))

    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Age", int(min_values["Age"]), int(max_values["Age"]))

    with col2:
        experience = st.slider(
            "Work Experience",
            int(min_values["Work_Experience"]),
            int(max_values["Work_Experience"]),
        )
    with col3:
        family_size = st.slider(
            "Family Size",
            int(min_values["Family_Size"]),
            int(max_values["Family_Size"]),
        )
    st.divider()

    pred_var = [
        [gender, married, age, graduated, profession, experience, spending, family_size]
    ]
    columns = [
        "Gender",
        "Ever_Married",
        "Age",
        "Graduated",
        "Profession",
        "Work_Experience",
        "Spending_Score",
        "Family_Size",
    ]
    pred_x = pd.DataFrame(pred_var, columns=columns)
    pred = predict(pred_x)
    st.write(f"Customer segmentation result : {pred}")

    st.divider()

    st.write("Improving by MEB")
    
    st.divider()


start_streamlit()
