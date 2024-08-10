import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.preprocessing import  OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import os

import warnings
warnings.filterwarnings('ignore') 

import pickle
import os

def pre_process_data():
    df = pd.read_csv('data/BankChurners.csv')
    df = df.iloc[:, :-2]
    client_num = df['CLIENTNUM']
    del df['CLIENTNUM']
    df=scale(df)

    return df

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


def skew(df,skew_lim=0.5):
    numeric = df.select_dtypes(exclude=object).columns
    skew_limit = skew_lim
    skew_vals = df[numeric].skew()

    skew_cols = (skew_vals
                .sort_values(ascending=False)
                .to_frame()
                .rename(columns={0:'Skew'})
                .query('abs(Skew) > {0}'.format(skew_limit)))
    for col in skew_cols.index:
        if (df[col] <= 0).any():
            df[col] = df[col] + 1  # Negatif ve sıfır değerleri düzelt
        try:
            lambda_ = boxcox_normmax(df[col])
            df[col] = boxcox1p(df[col], lambda_)
        except Exception as e:
            print(f"Kolon {col} için boxcox dönüşüm hatası: {e}")
    
    return df,numeric


def scale(df):
    df,numeric=skew(df)
    scalers={}
    for col in df[numeric]:
        scaler=MinMaxScaler()
        df[col] = scaler.fit_transform(df[[col]])
        scalers[col]=scaler

    
    return df



def encode(data):
    df_le = data.copy()
    df_le = pd.get_dummies(df_le, drop_first=False)  # drop_first=False, tüm dummy değişkenleri oluşturur
    return df_le


def predict(In_values):
    pred_data = load_and_transform(In_values)

    with open("models/knn_model.pkl", "rb") as file:
        model = pickle.load(file)
    prediction = model.predict(pred_data)
    return prediction[0]


def draw_graph():
    if os.path.exists("data/x_pca.csv"):
        X_pca=pd.read_csv("data/x_pca.csv")
    else: 
        df = pre_process_data()
        X = df.drop('Attrition_Flag_Attrited Customer', axis=1)
        y = df['Attrition_Flag_Attrited Customer']
        if os.path.exists("models/smote.pkl"):
            with open("models/smote.pkl", "rb") as file:
                smote = pickle.load(file)
        else:
            update_model()
            with open("models/smote.pkl", "rb") as file:
                smote = pickle.load(file)
        
        X_smote, y_smote = smote.fit_resample(X, y)
        y_smote.value_counts(normalize=True) * 100

        if os.path.exists("models/pca.pkl"):
            with open("models/pca.pkl", "rb") as file:
                pca = pickle.load(file)
        else:
            update_model()
            with open("models/pca.pkl", "rb") as file:
                pca = pickle.load(file)
        X_pca = pca.transform(X_smote)
        X_pca = pd.DataFrame(X_pca)
        


    km_list = list()

    for i in range(1,21):
        km = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
        km = km.fit(X_pca)
        
        km_list.append(pd.Series({'clusters': i, 
                                'inertia': km.inertia_,
                                'model': km}))     
    return km_list



def update_model():
    df = pre_process_data()
    df=encode(df)
    df= df[df.columns.drop(list(df.filter(regex='Unknown')))]
    df= df[df.columns.drop(list(df.filter(regex='Platinum')))]

    # Split target & features
    
    X = df.drop('Attrition_Flag_Attrited Customer', axis=1)
    y = df['Attrition_Flag_Attrited Customer']

    smote = SMOTE(random_state=42)
    
    X_smote, y_smote = smote.fit_resample(X, y)
    with open("models/smote.pkl", "wb") as file:
        pickle.dump(smote, file)
    
    y_smote.value_counts(normalize=True) * 100
    
    pca = PCA(.95) 
    pca.fit(X_smote)

    with open("models/pca.pkl", "wb") as file:
        pickle.dump(pca, file)

    X_pca = pca.transform(X_smote)
    X_pca = pd.DataFrame(X_pca)

    X_pca.to_csv('data/x_pca.csv')
    
    
    #The elbow method allows us to determine the optimal number of clusters
    # In this case, even though it will be ideal to have 4-8 clusters,
    #we'll keep it in two clusters to have the same number as the classes we are trying to predict: churn or not churn.
 
    
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y_smote, test_size=0.3, random_state=42)
    km = KMeans(n_clusters=2, random_state=42)
    km = km.fit(X_train)
    with open("models/k-means_model.pkl", "wb") as file:
        pickle.dump(km, file)

    y_pred=km.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1=f1_score(y_test, y_pred)
    model_status = [accuracy, f1]
    with open("models/model_performance.txt", "w") as file:
        for metric in model_status:
            file.write(f"{metric}\n")
    return km


"""    
    https://www.kaggle.com/code/rnakhi/experimenting-customer-segment-classification/notebook
"""
