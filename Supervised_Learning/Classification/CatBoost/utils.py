import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


import pickle
import os


def pre_process_data():
    data_test = pd.read_csv("data/Test.csv", sep=",")
    data_train = pd.read_csv("data/Train.csv", sep=",")

    # Concat both the data files
    data = pd.concat([data_test, data_train])
    data.drop("ID", inplace=True, axis=1)
    # ***___****
    custom_values = {
        "Ever_Married": "No",
        "Graduated": "No",
        "Profession": "Unemployed",
        "Work_Experience": 0,
        "Family_Size": 0,
    }
    df_filled = data.fillna(value=custom_values)
    # ***___****
    column_to_drop = "Var_1"
    df = df_filled.drop(column_to_drop, axis=1)
    # ***___****

    return df


def out_encode(data):
    le = LabelEncoder()
    encoded_data = le.fit_transform(data)
    with open("models/out_encoder.pkl", "wb") as file:
        pickle.dump(le, file)
    return encoded_data


def predict(In_values):
    pred_data = In_values
    with open("models/CatB_model.pkl", "rb") as file:
        model = pickle.load(file)
    prediction = model.predict(pred_data)
    with open("models/out_encoder.pkl", "rb") as file:
        out_encoder = pickle.load(file)
    prediction = out_encoder.inverse_transform(prediction)
    return prediction[0]


def update_model():
    df = pre_process_data()

    X = df.drop("Segmentation", axis=1)  # Features (excluding the target)
    y = df["Segmentation"]  # Target variable

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # save scaler model for after
    os.makedirs("models/", exist_ok=True)
    categorical_features = [
        "Gender",
        "Ever_Married",
        "Graduated",
        "Profession",
        "Spending_Score",
    ]

    y_test_encode = out_encode(y_test)
    y_train_encode = out_encode(y_train)

    catboost_model = CatBoostClassifier(
        iterations=100,
        learning_rate=0.1,
        depth=6,
        cat_features=categorical_features,
        verbose=0,
    )
    catboost_model.fit(X_train, y_train_encode)
    with open("models/CatB_model.pkl", "wb") as file:
        pickle.dump(catboost_model, file)

    y_pred = catboost_model.predict(X_test)
    accuracy = accuracy_score(y_test_encode, y_pred)
    report = classification_report(y_test_encode, y_pred)
    model_status = [accuracy, report]
    with open("models/model_performance.txt", "w") as file:
        for metric in model_status:
            file.write(f"{metric}\n")
    return catboost_model


"""    
    https://www.kaggle.com/code/rnakhi/experimenting-customer-segment-classification/notebook
"""
