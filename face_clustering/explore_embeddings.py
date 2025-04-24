import pandas as pd
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def collect_data(model_name: str) -> pd.DataFrame:
    """

    """
    # Create connection to the database
    database_file = "face_embeddings/data.db"
    conn = sqlite3.connect(database_file)

    q = f"""
    SELECT m.id_path, embedding, embedding_pca, embedding_tsne, person
    FROM {model_name} AS m, persons AS p
    WHERE m.id_path == p.id_path
    """

    df = pd.read_sql_query(q, conn)
    df["embedding"] = df["embedding"].apply(lambda x: np.fromstring(x, sep=","))
    df["embedding_tsne"] = df["embedding_tsne"].apply(lambda x: np.fromstring(x, sep=","))
    df["embedding_pca"] = df["embedding_pca"].apply(lambda x: np.fromstring(x, sep=","))

    return df


def plot_cluster(data: pd.DataFrame, model: str, pca_or_tsne: str) -> None:
    """

    """
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=data,
        x="x",
        y="y",
        hue="person",
        hue_order=sorted(data["person"].unique()),
        palette=sns.color_palette("tab20", n_colors=len(data["person"].unique())),
        s=30
    )

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Person")
    plt.title(f"{pca_or_tsne} of Face Embeddings by model {model}")
    plt.tight_layout()

    if not os.path.exists(f"plots/{pca_or_tsne}"):
        os.makedirs(f"plots/{pca_or_tsne}")
    plt.savefig(f"plots/{pca_or_tsne}/{model}.jpg")


def main_plot_embeddings() -> None:
    for p in os.listdir("face_embeddings"):
        if p.endswith(".npz"):
            model = "_".join(p.split("_")[:-1]).replace("-", "_")

            df = collect_data(model)

            df["x"] = df["embedding_tsne"].apply(lambda lst: lst[0])
            df["y"] = df["embedding_tsne"].apply(lambda lst: lst[1])

            plot_cluster(df, p, "t-SNE")

            df["x"] = df["embedding_pca"].apply(lambda lst: lst[0])
            df["y"] = df["embedding_pca"].apply(lambda lst: lst[1])

            plot_cluster(df, p, "PCA")


if __name__ == "__main__":
    main_plot_embeddings()
