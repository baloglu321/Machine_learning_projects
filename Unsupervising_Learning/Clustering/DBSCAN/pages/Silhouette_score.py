import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from time import sleep
from utils import *
import warnings

warnings.filterwarnings("ignore")

# Streamlit UI
st.title("DBSCAN için silhouette score kullanarak küme sayısı tavsiye uygulaması")

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

    if st.button("Silhouette score hesapla"):
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

                    pca = PCA(n_components=2)
                    X_pca = pca.fit_transform(X_train)
                    X_pca = pd.DataFrame(X_pca, columns=["PCA1", "PCA2"])
                    eps_values = np.linspace(0.1, 1.0, 10)
                    min_samples_range = range(3, 11)
                    num_clusters = []

                    best_score = -1
                    best_params = None
                    progress = 40

                    for eps in eps_values:
                        progress = progress + (20 // len(eps_values))
                        progress_bar.progress(progress)

                        cluster_counts = []
                        for min_samples in min_samples_range:
                            db = DBSCAN(eps=eps, min_samples=min_samples)
                            labels = db.fit_predict(X_train)

                            if len(set(labels)) > 1:
                                score = silhouette_score(X_train, labels)
                                if score > best_score:
                                    best_score = score
                                    best_params = (eps, min_samples)

                            cluster_count = len(set(labels)) - (
                                1 if -1 in labels else 0
                            )
                            cluster_counts.append(cluster_count)

                        num_clusters.append(np.mean(cluster_counts))

                    # Grafik oluşturma
                    plt.figure(figsize=(12, 8))
                    plt.plot(eps_values, num_clusters, marker="o", color="#00538F")
                    plt.xlabel("eps")
                    plt.ylabel("Number of Clusters")
                    plt.title("DBSCAN: Number of Clusters vs eps")
                    plt.grid(True)

                    st.pyplot(plt)
                    progress_bar.progress(60)
                    sleep(1)
                    st.title(
                        f"En iyi parametreler: eps={best_params[0]}, min_samples={best_params[1]}"
                    )
                    progress_bar.progress(80)
                    sleep(1)  # Simulate processing time

                else:
                    st.error("İşlenmesi gereken kolon seçilmelidir!")
        else:
            st.error("Ana kolon seçilmelidir!")

        progress_bar.empty()
