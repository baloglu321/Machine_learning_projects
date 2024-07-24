import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


import pickle
import os


def load_and_transform(data):
    with open("models/label_encoder.pkl", "rb") as file:
        encoders = pickle.load(file)
    df_le = data.copy()
    c = df_le.dtypes == "object"
    object_cols = list(c[c].index)

    for i in object_cols:
        if i in encoders:
            encoder = encoders[i]
            df_le[i] = encoder.transform(df_le[i])
        else:
            raise ValueError(f"No encoder found for column {i}")
    with open("models/scaler.pkl", "rb") as file:
        scaler = pickle.load(file)
    df_le = scaler.transform(df_le)
    return df_le


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


def encode(data):
    encoders = {}
    df_le = data.copy()
    c = df_le.dtypes == "object"
    object_cols = list(c[c].index)
    object_cols.remove("Segmentation")

    for i in object_cols:
        le = LabelEncoder()
        df_le[i] = le.fit_transform(df_le[i])
        encoders[i] = le

    with open("models/label_encoder.pkl", "wb") as file:
        pickle.dump(encoders, file)

    return df_le


def predict(In_values):
    pred_data = load_and_transform(In_values)

    with open("models/bayes_model.pkl", "rb") as file:
        model = pickle.load(file)
    prediction = model.predict(pred_data)
    return prediction[0]


def update_model():
    df = pre_process_data()

    encode_df = encode(df)

    X = encode_df.drop("Segmentation", axis=1)  # Features (excluding the target)
    y = encode_df["Segmentation"]  # Target variable

    # Split the data into training and testing sets
    X_train_LE, X_test_LE, y_train_LE, y_test_LE = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_LE)
    X_test_scaled = scaler.transform(X_test_LE)
    # save scaler model for after
    os.makedirs("models/", exist_ok=True)
    with open("models/scaler.pkl", "wb") as file:
        pickle.dump(scaler, file)

    model_naive_bayes = GaussianNB()
    model_naive_bayes.fit(X_train_scaled, y_train_LE)
    with open("models/bayes_model.pkl", "wb") as file:
        pickle.dump(model_naive_bayes, file)

    y_pred = model_naive_bayes.predict(X_test_scaled)
    accuracy = accuracy_score(y_test_LE, y_pred)
    report = classification_report(y_test_LE, y_pred)
    model_status = [accuracy, report]
    with open("models/model_performance.txt", "w") as file:
        for metric in model_status:
            file.write(f"{metric}\n")
    return model_naive_bayes


"""    
    https://www.kaggle.com/code/rnakhi/experimenting-customer-segment-classification/notebook
"""