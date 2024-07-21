import datetime
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime
import pickle
import os


def load_and_transform(data):
    with open("models/label_encoder.pkl", "rb") as file:
        encoders = pickle.load(file)

    numeric_columns = data.select_dtypes(include="number")
    string_columns = data.select_dtypes(include="object")

    for col in string_columns.columns:
        if col in encoders:
            encoder = encoders[col]
            string_columns[col] = encoder.transform(string_columns[col])
        else:
            raise ValueError(f"No encoder found for column {col}")

    data = pd.concat([string_columns, numeric_columns], axis=1)
    return data


def convert_str2_num(data):
    encoders = {}
    numeric_columns = data.select_dtypes(include="number")
    string_columns = data.select_dtypes(include="object")

    for col in string_columns.columns:
        la = LabelEncoder()
        string_columns[col] = la.fit_transform(string_columns[col])
        encoders[col] = la

    data = pd.concat([string_columns, numeric_columns], axis=1)
    for col in data.columns:
        if data[col].dtype == "object":
            try:
                data[col] = pd.to_numeric(data[col])
            except ValueError:
                pass  # If conversion fails, leave the column as object
    # save la model for after
    os.makedirs("models/", exist_ok=True)
    with open("models/label_encoder.pkl", "wb") as file:
        pickle.dump(encoders, file)

    # Detect Outliers
    for col in numeric_columns.columns:
        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        outlier = ((numeric_columns[col] > high) | (numeric_columns[col] < low)).sum()
        if outlier > 0:
            data = data.loc[(data[col] <= high) & (data[col] >= low)]

    return data


def pre_process_data():
    df = pd.read_csv("data/car_price_prediction.csv")  # read the data
    # remove duplicate data
    df.drop_duplicates(inplace=True)
    # calc car age
    Dtime = datetime.now()
    df["Age"] = Dtime.year - df["Prod. year"]
    # regulate blank levy data
    df.Levy.replace({"-": "0"}, inplace=True)
    df["Levy"] = df["Levy"].astype(int)
    # Remove str km
    df["Mileage"] = df["Mileage"].str.replace("km", "")
    # data_info=df.info()
    # Regulate engine volume
    df["Engine volume"] = df["Engine volume"].str.replace("Turbo", "")
    df["Engine volume"] = df["Engine volume"].astype(float)

    # remove unnecessary columns

    raw_data = df.drop(["ID", "Doors", "Prod. year"], axis=1)
    for col in raw_data.columns:
        if raw_data[col].dtype == "object":
            try:
                raw_data[col] = pd.to_numeric(raw_data[col])
            except ValueError:
                pass  # If conversion fails, leave the column as object
    # Convert verbal data to numerical data
    data_num = convert_str2_num(raw_data)

    return df, data_num, raw_data


def check_is_dir(path):
    if not os.path.isdir(path):
        return False
    else:
        return True


def convert_pred_data(data):
    if check_is_dir("models/"):
        encoded_labels = load_and_transform(data)
        with open("models/scaler.pkl", "rb") as file:
            scaler = pickle.load(file)
        scale_data = scaler.transform(encoded_labels)
        with open("models/poly_transformer.pkl", "rb") as file:
            poly = pickle.load(file)
        poly_data = poly.transform(scale_data)
    else:
        update_model()
        encoded_labels = load_and_transform(data)
        with open("models/scaler.pkl", "rb") as file:
            scaler = pickle.load(file)
        scale_data = scaler.transform(encoded_labels)
        with open("models/poly_transformer.pkl", "rb") as file:
            poly = pickle.load(file)
        poly_data = poly.transform(scale_data)

    return poly_data


def predict_price(In_values):
    poly_data = convert_pred_data(In_values)
    with open("models/poly_model.pkl", "rb") as file:
        model = pickle.load(file)
    prediction = model.predict(poly_data)
    return float(prediction[0])


def update_model():
    df, data, raw_data = pre_process_data()

    X = data.drop(
        "Price", axis=1
    )  # Mean number of room, percentage of population with low status,average number of people
    y = data["Price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # save scaler model for after
    os.makedirs("models/", exist_ok=True)
    with open("models/scaler.pkl", "wb") as file:
        pickle.dump(scaler, file)

    degree = 3
    poly = PolynomialFeatures(degree)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)
    # save transform model for after
    with open("models/poly_transformer.pkl", "wb") as file:
        pickle.dump(poly, file)
    # save model for after
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    with open("models/poly_model.pkl", "wb") as file:
        pickle.dump(model, file)

    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    model_status = [train_mse, test_mse, train_r2, test_r2]
    with open("models/model_performance.txt", "w") as file:
        for metric in model_status:
            file.write(f"{metric}\n")
    return model
