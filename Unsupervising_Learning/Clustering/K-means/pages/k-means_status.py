import streamlit as st
import pandas as pd

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib as mpl
from time import sleep
from sklearn.decomposition import PCA
from utils import *
import warnings

warnings.filterwarnings("ignore")

# Streamlit UI
st.title("K-Means Kümeleme Uygulaması")

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
    num_clusters = st.slider(
        "Max küme sayısını seçin", min_value=1, max_value=20, value=3
    )

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

                    pca = PCA(n_components=2)
                    X_pca = pca.fit_transform(X_train)
                    X_pca = pd.DataFrame(X_pca, columns=["PCA1", "PCA2"])

                    progress_bar.progress(80)
                    sleep(1)  # Simulate processing time

                    km_list = list()

                    for i in range(1, num_clusters):
                        km = KMeans(
                            n_clusters=i,
                            init="k-means++",
                            max_iter=300,
                            n_init=10,
                            random_state=42,
                        )
                        km = km.fit(X_pca)

                        km_list.append(
                            pd.Series(
                                {"clusters": i, "inertia": km.inertia_, "model": km}
                            )
                        )

                    k = pd.concat(km_list, axis=1).T[["clusters", "inertia"]]

                    # Visualize
                    fig, ax = plt.subplots(figsize=(12, 8))
                    fig.patch.set_facecolor("white")
                    mpl.rcParams["font.size"] = 14

                    plt.plot(k["clusters"], k["inertia"], "bo-", color="#00538F")

                    # Remove ticks
                    ax.xaxis.set_ticks_position("none")
                    ax.yaxis.set_ticks_position("none")

                    # Remove axes splines
                    for i in ["top", "right"]:
                        ax.spines[i].set_visible(False)

                    ax.set_xticks(range(0, 21, 2))
                    ax.set(xlabel="Cluster", ylabel="Inertia")

                    plt.suptitle(
                        "The Elbow Method: Optimal Number of Clusters", size=26
                    )

                    st.pyplot(fig)

                else:
                    st.error("İşlenmesi gereken kolon seçilmelidir!")
        else:
            st.error("Ana kolon seçilmelidir!")

        progress_bar.empty()  # İlerleme çubuğunu temizle
