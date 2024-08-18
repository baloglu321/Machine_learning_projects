import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from time import sleep
from utils import *
import warnings

warnings.filterwarnings("ignore")

# Streamlit UI
st.title(
    "Hierarchical Clustering için elbow methodu kullanarak küme sayısı tavsiye uygulaması"
)

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

                    pca = PCA(n_components=2)
                    X_pca = pca.fit_transform(X_train)
                    X_pca = pd.DataFrame(X_pca, columns=["PCA1", "PCA2"])

                    progress_bar.progress(60)
                    sleep(1)
                    progress = 60

                    hc_list = list()

                    for i in range(2, num_clusters):
                        progress_bar.progress(progress)
                        progress = progress + (20 // num_clusters)
                        hc = AgglomerativeClustering(
                            n_clusters=i, affinity="euclidean", linkage="ward"
                        )
                        hc = hc.fit(X_pca)

                        hc_list.append(
                            pd.Series(
                                {
                                    "clusters": i,
                                    "linkage_distance": sch.linkage(
                                        X_pca, method="ward"
                                    )[i, 2],
                                }
                            )
                        )

                    k = pd.concat(hc_list, axis=1).T[["clusters", "linkage_distance"]]
                    first_derivative = np.diff(k["linkage_distance"])

                    second_derivative = np.diff(first_derivative)

                    elbow_point = (
                        np.argmax(second_derivative) + 2
                    )  # +2, indeks düzeltmesi

                    fig, ax = plt.subplots(figsize=(12, 8))
                    fig.patch.set_facecolor("white")

                    plt.plot(
                        k["clusters"],
                        k["linkage_distance"],
                        "bo-",
                        color="#00538F",
                        label="Linkage Distance",
                    )
                    plt.axvline(
                        x=elbow_point,
                        color="r",
                        linestyle="--",
                        label=f"Elbow Point: {elbow_point} clusters",
                    )

                    ax.xaxis.set_ticks_position("none")
                    ax.yaxis.set_ticks_position("none")

                    for i in ["top", "right"]:
                        ax.spines[i].set_visible(False)

                    ax.set_xticks(range(0, num_clusters, 2))
                    ax.set(xlabel="Cluster", ylabel="Linkage Distance")

                    plt.suptitle(
                        "The Elbow Method for Hierarchical Clustering", size=26
                    )
                    plt.legend()
                    st.pyplot(fig)

                    progress_bar.progress(80)
                    sleep(1)  # Simulate processing time

                else:
                    st.error("İşlenmesi gereken kolon seçilmelidir!")
        else:
            st.error("Ana kolon seçilmelidir!")

        progress_bar.empty()
