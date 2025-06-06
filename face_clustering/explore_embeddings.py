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
    SELECT m.id_path, embedding, embedding_tsne, person
    FROM {model_name} AS m, persons AS p
    WHERE m.id_path == p.id_path
    """

    # Query the SQL database
    df = pd.read_sql_query(q, conn)

    # Set embeddings to preferred format
    df["embedding"] = df["embedding"].apply(lambda x: np.fromstring(x, sep=","))
    df["embedding_tsne"] = df["embedding_tsne"].apply(lambda x: np.fromstring(x, sep=","))

    return df


def plot_cluster_combined(axs, data: pd.DataFrame, index: int):
    """
    Plots the embeddings on an axes in a figure such that the left plot is for the CFI dataset and
    the right plot is for the Embedding on the Wall dataset.
    """
    # Plot the data
    sns.scatterplot(
        data=data,
        x="x",
        y="y",
        hue="person",
        hue_order=sorted(data["person"].unique()),
        palette=sns.color_palette("tab20", n_colors=len(data["person"].unique())),
        s=30,
        ax=axs[index]
    )

    # Remove the legend
    axs[index].legend().remove()

    # Add titles for each plot
    if index == 0:
        axs[index].set_title(f"The CFI dataset", fontsize=20)
    else:
        axs[index].set_title("The 'Embeddings on the Wall' dataset", fontsize=20)

    # Update figure layout
    axs[index].set_xlabel("x", fontsize=16)
    axs[index].set_ylabel("y", fontsize=16)
    axs[index].tick_params(axis="both", which="major", labelsize=14)

    return axs


def main_plot_embeddings_combined() -> None:
    """
    Main runner for plotting all embeddings using all models
    for the CFI dataset and the Embedding on the Wall dataset.
    """
    # Check if save folder exists
    if not os.path.exists("plots"):
        os.mkdir("plots")

    # Set all models to plot
    models = ["ArcFace", "DeepFace", "Dlib", "FaceNet", "face_recognition", "OpenFace", "VGG_Face"]

    # Iterate over all models
    for model in models:
        # Load the embeddings for both datasets
        df = collect_data(model)
        df_wies = collect_data(f"Wies_{model}")

        # Set the x and y for the t-SNE reduced embeddings
        df["x"] = df["embedding_tsne"].apply(lambda lst: lst[0])
        df["y"] = df["embedding_tsne"].apply(lambda lst: lst[1])
        df_wies["x"] = df_wies["embedding_tsne"].apply(lambda lst: lst[0])
        df_wies["y"] = df_wies["embedding_tsne"].apply(lambda lst: lst[1])

        # Initialize the figure
        fig, axs = plt.subplots(1, 2, figsize=(18, 8), sharex=False, sharey=False)
        plt.suptitle(f"t-SNE reduced face embedding, embedded using model: '{model}'", fontsize=20, weight="bold")

        # Plot the embeddings
        axs = plot_cluster_combined(axs, df.copy(), 0)
        axs = plot_cluster_combined(axs, df_wies.copy(), 1)
        plt.tight_layout()

        # Save the figure
        plt.savefig(f"plots/combined_{model}.pdf")


if __name__ == "__main__":
    main_plot_embeddings_combined()
