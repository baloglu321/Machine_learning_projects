import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split


import pickle
import os


def load_and_transform(data):
    with open("models/ohe_encoder.pkl", "rb") as file:
        encoders = pickle.load(file)
    df_le = data.copy()
    c = df_le.dtypes == "object"
    object_cols = list(c[c].index)

    for i in object_cols:
        if i in encoders:
            encoder = encoders[i]
            reshaped_data = df_le[i].values.reshape(-1, 1)
            df_ohe = encoder.transform(reshaped_data)
            ohe_df = pd.DataFrame(
                df_ohe, columns=encoder.get_feature_names_out([i]), index=df_le.index
            )
            df_le = df_le.drop(columns=[i])
            df_le = pd.concat([df_le, ohe_df], axis=1)

        else:
            raise ValueError(f"No encoder found for column {i}")

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
        ohe = OneHotEncoder(sparse=False)
        reshaped_data = df_le[i].values.reshape(-1, 1)
        df_ohe = ohe.fit_transform(reshaped_data)
        ohe_df = pd.DataFrame(
            df_ohe, columns=ohe.get_feature_names_out([i]), index=df_le.index
        )
        df_le = df_le.drop(columns=[i])
        df_le = pd.concat([df_le, ohe_df], axis=1)
        encoders[i] = ohe

    with open("models/ohe_encoder.pkl", "wb") as file:
        pickle.dump(encoders, file)

    return df_le


def out_encode(data):
    le = LabelEncoder()
    encoded_data = le.fit_transform(data)
    with open("models/out_encoder.pkl", "wb") as file:
        pickle.dump(le, file)
    return encoded_data


def predict(In_values):
    pred_data = load_and_transform(In_values)

    with open("models/adaboost_model.pkl", "rb") as file:
        model = pickle.load(file)
    prediction = model.predict(pred_data)
    with open("models/out_encoder.pkl", "rb") as file:
        out_encoder = pickle.load(file)
    prediction = out_encoder.inverse_transform(prediction)
    return prediction[0]


def update_model():
    df = pre_process_data()

    encode_df = encode(df)

    X = encode_df.drop("Segmentation", axis=1)  # Features (excluding the target)
    y = encode_df["Segmentation"]  # Target variable

    # Split the data into training and testing sets
    X_train_OHE, X_test_OHE, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # save scaler model for after
    os.makedirs("models/", exist_ok=True)

    y_test_encode = out_encode(y_test)
    y_train_encode = out_encode(y_train)

    base_estimator = DecisionTreeClassifier(max_depth=1)
    adaboost = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=50, learning_rate=1.0, random_state=42)
    adaboost.fit(X_train_OHE, y_train_encode)
    with open("models/adaboost_model.pkl", "wb") as file:
        pickle.dump(adaboost, file)

    y_pred = adaboost.predict(X_test_OHE)
    accuracy = accuracy_score(y_test_encode, y_pred)
    report = classification_report(y_test_encode, y_pred)
    model_status = [accuracy, report]
    with open("models/model_performance.txt", "w") as file:
        for metric in model_status:
            file.write(f"{metric}\n")
    return adaboost


"""    
    https://www.kaggle.com/code/rnakhi/experimenting-customer-segment-classification/notebook
"""
