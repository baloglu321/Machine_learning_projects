import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from utils import *

# Streamlit UI
st.title('K-Means Kümeleme Uygulaması')

# Upload Data
uploaded_file = st.file_uploader("Veri dosyasını yükleyin (CSV formatında)", type=["csv"])
if uploaded_file is not None:
    # Upload Data
    df = pd.read_csv(uploaded_file)
    st.write("Yüklenen veri:")
    st.dataframe(df.head())
    
    # Show Columns
    columns = df.columns
    st.write("Mevcut kolonlar:")
    st.write(columns)

    # Chose main column
    main_column = st.selectbox("Ana kümeleme kolonu seçin", columns)
    new_columns=columns.drop(main_column)

    # Chose other columns
    processing_columns = st.multiselect("İşlenmesini istediğiniz kolonları seçin", new_columns, default=new_columns.tolist())
    
    # Chose maximum cluster size
    num_clusters = st.slider("Max küme sayısını seçin", min_value=1, max_value=20, value=3)
    
    if st.button("Model oluştur"):
        if main_column:
            # Check main column
            if not processing_columns:
                st.error("İşlenmesi gereken en az bir kolon seçilmelidir!")
            else:
                # process data
                processing_columns = [col for col in processing_columns if col != main_column]
                
                if processing_columns:
                                        
                    X = df[processing_columns]
                    y = pd.DataFrame(df[main_column])
                    

                    X_train=scale(X)
                    y_train=scale(y)

                    X_train=encode(X_train)
                    y_train=encode(y_train)

                    pca = PCA(.95) 
                    pca.fit(X_train)


                    X_pca = pca.transform(X_train)
                    X_pca = pd.DataFrame(X_pca)

                    
                    # K-Means kümeleme
                    kmeans = KMeans(n_clusters=num_clusters)
                    X['Cluster'] = kmeans.fit_predict(X_pca)
                    
                    # Sonuçları göster
                    st.write("Kümeleme sonuçları:")
                    st.dataframe(X)
                    
                    # Küme merkezlerini göster
                    st.write("Küme merkezleri:")
                    st.write(kmeans.cluster_centers_)
                else:
                    st.error("İşlenmesi gereken kolon seçilmelidir!")
        else:
            st.error("Ana kolon seçilmelidir!")


