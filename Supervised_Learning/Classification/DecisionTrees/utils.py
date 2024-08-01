import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle


def train_decision_tree(
    X_train, X_test, y_train, criterion, max_depth, min_samples_leaf
):
    # Creating the classifier object
    clf = DecisionTreeClassifier(
        criterion=criterion,
        random_state=100,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
    )

    # Performing training
    clf.fit(X_train, y_train)
    return clf


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

    with open("models/decision_model.pkl", "rb") as file:
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

    clf_gini = train_decision_tree(X_train_LE, X_test_LE, y_train_LE, "gini", 3, 5)
    with open("models/decision_model.pkl", "wb") as file:
        pickle.dump(clf_gini, file)

    y_pred = clf_gini.predict(X_test_LE)
    accuracy = accuracy_score(y_test_LE, y_pred)
    report = classification_report(y_test_LE, y_pred)
    model_status = [accuracy, report]
    with open("models/model_performance.txt", "w") as file:
        for metric in model_status:
            file.write(f"{metric}\n")
    return clf_gini


"""    
    https://www.kaggle.com/code/rnakhi/experimenting-customer-segment-classification/notebook
"""
