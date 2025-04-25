from typing import Tuple
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import DBSCAN
from explore_embeddings import collect_data
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import fowlkes_mallows_score
import openpyxl  # Needed for writing xlsx files
from tqdm import tqdm


def check_performance_dbscan(df: pd.DataFrame) -> Tuple[float, float]:
    """
    Uses performance metrics to check the performance or clustering algorithms.

    :param df: dataframe containing the actual and predicted clusters

    :return: the adjusted rand score and fowlkes mallows score
    """
    # Similarity between two clustering algorithms
    ari = adjusted_rand_score(df["person"], df["cluster"])

    # sqrt(precision*recall)
    fmi = fowlkes_mallows_score(df["person"], df["cluster"])

    return ari, fmi


def main_test_embeddings_dbscan() -> None:
    """
    Main runner for testing different embeddings with different parameters for clustering.
    """
    # Get all parameters to use
    ebsilons = [i*0.1 for i in range(1, 101)]

    # Iterables to create df with
    models = ["ArcFace", "Dlib", "Facenet", "face_recognition", "VGG_Face"]
    dbscan_params = ["eps"]
    metrics = ["ARI", "FMI", "rel"]

    # Create multi-indexed columns for dataframe
    multi_cols = pd.MultiIndex.from_product([models, metrics])
    all_columns = pd.MultiIndex.from_tuples([("params", p) for p in dbscan_params] + list(multi_cols))
    result = pd.DataFrame(columns=all_columns)

    # Initialize progressbar
    pbar = tqdm(total=len(ebsilons), ncols=100, desc="Checking all possible parameters")

    # Iterate over all parameters
    for ebs in ebsilons:
        # append parameter to the result
        res = [ebs]

        # cluster using different embedding models
        for model in models:
            # Collect the embeddings
            data = collect_data(model)

            # Cluster based on the embeddings with the set parameters
            labels = DBSCAN(eps=ebs, min_samples=1, n_jobs=-1, metric="euclidean").fit_predict(data["embedding_tsne"].tolist())

            # Add the clusters to the data
            data1 = data.copy()
            data1["cluster"] = labels

            # Collect the performance metrics
            ari, fmi = check_performance_dbscan(data1.copy())

            # Append the performance metrics to the result
            res.append(ari)
            res.append(fmi)
            res.append(len(np.unique(labels)) - 17)

        # Add the resulting row to the final dataframe
        result.loc[len(result)] = [round(i, 3) for i in res]

        # Update the progressbar
        pbar.update(1)

    # Save the result
    result.to_excel("results_dbscan_parameters.xlsx")


if __name__ == "__main__":
    main_test_embeddings_dbscan()
