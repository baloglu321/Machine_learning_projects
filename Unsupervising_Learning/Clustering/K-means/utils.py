import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax


import warnings

warnings.filterwarnings("ignore")


def skew(df, skew_lim=0.5):
    numeric = df.select_dtypes(exclude=object).columns
    skew_limit = skew_lim
    skew_vals = df[numeric].skew()

    skew_cols = (
        skew_vals.sort_values(ascending=False)
        .to_frame()
        .rename(columns={0: "Skew"})
        .query("abs(Skew) > {0}".format(skew_limit))
    )
    for col in skew_cols.index:
        if (df[col] <= 0).any():
            df[col] = df[col] + 1
        try:
            lambda_ = boxcox_normmax(df[col])
            df[col] = boxcox1p(df[col], lambda_)
        except Exception as e:
            print(f"Kolon {col} için boxcox dönüşüm hatası: {e}")

    return df, numeric


def scale(df):
    df, numeric = skew(df)
    scalers = {}
    for col in df[numeric]:
        scaler = MinMaxScaler()
        df[col] = scaler.fit_transform(df[[col]])
        scalers[col] = scaler

    return df


def encode(data):
    df_le = data.copy()
    df_le = pd.get_dummies(df_le, drop_first=False)
    return df_le
