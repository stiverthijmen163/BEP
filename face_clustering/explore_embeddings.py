import pandas as pd
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def collect_data(model_name: str) -> pd.DataFrame:
    """
    Collects the data from the database with the corresponding model used for calculating the embeddings.

    :param model_name: the name of the model used for calculating the embeddings

    :return: the collected data in a dataframe
    """
    # Create connection to the database
    database_file = "face_embeddings/data.db"
    conn = sqlite3.connect(database_file)

    # Create query
    q = f"""
    SELECT m.id_path, embedding, embedding_pca, embedding_tsne, person
    FROM {model_name} AS m, persons AS p
    WHERE m.id_path == p.id_path
    """

    # Query the SQL database
    df = pd.read_sql_query(q, conn)

    # Set embeddings to preferred format
    df["embedding"] = df["embedding"].apply(lambda x: np.fromstring(x, sep=","))
    df["embedding_tsne"] = df["embedding_tsne"].apply(lambda x: np.fromstring(x, sep=","))
    df["embedding_pca"] = df["embedding_pca"].apply(lambda x: np.fromstring(x, sep=","))

    return df


def plot_cluster(data: pd.DataFrame, model: str, pca_or_tsne: str) -> None:
    """
    Plots the embeddings and saves this plot.

    :param data: dataframe containing the data to be plotted, must contain x, y and person
    :param model: the name of the model used for calculating the embeddings
    :param pca_or_tsne: whether the embeddings are reduced using PCA or t-SNE
    """
    # Initialize the figure
    plt.figure(figsize=(10, 8))

    # Plot the data
    sns.scatterplot(
        data=data,
        x="x",
        y="y",
        hue="person",
        hue_order=sorted(data["person"].unique()),
        palette=sns.color_palette("tab20", n_colors=len(data["person"].unique())),
        s=30
    )

    # Adjust the figure as preferred
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Person")
    plt.title(f"{pca_or_tsne} of Face Embeddings by model {model}")
    plt.tight_layout()

    # Create save path if needed
    if not os.path.exists(f"plots/{pca_or_tsne}"):
        os.makedirs(f"plots/{pca_or_tsne}")

    # Save the figure
    plt.savefig(f"plots/{pca_or_tsne}/{model}.jpg")


def main_plot_embeddings() -> None:
    """
    Main runner for plotting all different embeddings.
    """
    # Iterate over all models
    for p in os.listdir("face_embeddings"):
        if p.endswith(".npz"):
            # Set the model's name
            model = "_".join(p.split("_")[:-1]).replace("-", "_")

            # Collect the embeddings
            df = collect_data(model)

            # Set the x and y for the t-SNE reduced embeddings
            df["x"] = df["embedding_tsne"].apply(lambda lst: lst[0])
            df["y"] = df["embedding_tsne"].apply(lambda lst: lst[1])

            # Plot the t-SNE reduced embeddings
            plot_cluster(df, p, "t-SNE")

            # Set the x and y for the PCA reduced embeddings
            df["x"] = df["embedding_pca"].apply(lambda lst: lst[0])
            df["y"] = df["embedding_pca"].apply(lambda lst: lst[1])

            # Plot the PCA reduced embeddings
            plot_cluster(df, p, "PCA")


if __name__ == "__main__":
    main_plot_embeddings()
