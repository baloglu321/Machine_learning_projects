from utils import pre_process_data,predict_price
import streamlit as st



def start_streamlit():
    df,data_num,raw_data=pre_process_data()

    string_columns = raw_data.select_dtypes(include='object')
    numeric_columns = raw_data.select_dtypes(include='number')
    string_unq = {col: sorted(string_columns[col].unique()) for col in string_columns.columns}
    col1, col2,col3 = st.columns(3)
    with col1:
        brand = st.selectbox(
        "Car Brand",    tuple(string_unq['Manufacturer']))
        leather=st.selectbox(
        "Leather",    tuple(string_unq['Leather interior']))
        drive_whells=st.selectbox(
        'Drive wheels',    tuple(string_unq['Drive wheels']))
    with col2:
        car_model=st.selectbox(
        "Car Model",    tuple(string_unq['Model']))
        fuel_type=st.selectbox(
        "Fuel Type",    tuple(string_unq['Fuel type']))
        whell=st.selectbox(
        'Wheel',    tuple(string_unq['Wheel']))
    with col3:
        category=st.selectbox(
        "Car Category",    tuple(string_unq['Category']))
        gear_type=st.selectbox(
        "Gear box type",    tuple(string_unq['Gear box type']))
        color= st.selectbox(
        'Color',    tuple(string_unq['Color']))

        
    
start_streamlit()