import streamlit as st
import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from utils import *
from time import sleep
import warnings

warnings.filterwarnings("ignore")

# Streamlit UI
st.title("DBSCAN Uygulaması")

# Upload Data
uploaded_file = st.file_uploader(
    "Veri dosyasını yükleyin (CSV formatında)", type=["csv"]
)
if uploaded_file is not None:
    # Upload Data
    df = pd.read_csv(uploaded_file)
    st.write("Yüklenen veri:")
    st.dataframe(df.head())
    duplicate_rows = df.duplicated()
    df = df.drop_duplicates()
    df.dropna(inplace=True)
    # Show Columns
    columns = df.columns
    st.write("Mevcut kolonlar:")
    st.write(columns)

    # Chose main column
    main_column = st.selectbox("Ana kümeleme kolonu seçin", columns)
    new_columns = columns.drop(main_column)

    # Chose other columns
    processing_columns = st.multiselect(
        "İşlenmesini istediğiniz kolonları seçin",
        new_columns,
        default=new_columns.tolist(),
    )

    # Chose maximum cluster size
    num_clusters = st.slider("Küme sayısını seçin", min_value=1, max_value=20, value=3)
    eps = st.slider("Mesafe yarıçapını seçin", min_value=0.1, max_value=10.0, value=0.5)

    if st.button("Model oluştur"):
        progress_bar = st.progress(0)
        if main_column:
            # Check main column
            if not processing_columns:
                st.error("İşlenmesi gereken en az bir kolon seçilmelidir!")
            else:
                progress_bar.progress(20)
                sleep(1)  # Simulate processing time

                # process data
                processing_columns = [
                    col for col in processing_columns if col != main_column
                ]

                if processing_columns:
                    X = df[processing_columns]
                    y = pd.DataFrame(df[main_column])

                    progress_bar.progress(40)
                    sleep(1)  # Simulate processing time

                    X_train = scale(X)
                    y_train = scale(y)

                    X_train = encode(X_train)
                    y_train = encode(y_train)

                    progress_bar.progress(60)

                    pca = PCA(n_components=num_clusters)
                    pca.fit(X_train)

                    pca = PCA(n_components=2)
                    X_pca = pca.fit_transform(X_train)
                    X_pca = pd.DataFrame(X_pca, columns=["PCA1", "PCA2"])

                    progress_bar.progress(80)
                    sleep(1)  # Simulate processing time

                    dbscan = DBSCAN(eps=eps, min_samples=num_clusters)
                    clusters = dbscan.fit_predict(X_train)

                    # show the results

                    df = pd.DataFrame(X_pca, columns=["PCA1", "PCA2"])
                    df["Cluster"] = clusters

                    # update cluster
                    plt.figure(figsize=(10, 7))
                    sns.scatterplot(
                        data=df,
                        x="PCA1",
                        y="PCA2",
                        hue="Cluster",
                        palette="Set1",
                        s=100,
                        legend="full",
                    )
                    plt.title("DBSCAN Clustering on this Dataset")

                    st.pyplot(plt)
                else:
                    st.error("İşlenmesi gereken kolon seçilmelidir!")
        else:
            st.error("Ana kolon seçilmelidir!")
        progress_bar.empty()