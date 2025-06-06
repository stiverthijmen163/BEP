import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import DBSCAN
from explore_embeddings import collect_data
from sklearn.metrics import adjusted_rand_score
import openpyxl  # Needed for writing xlsx files
from tqdm import tqdm
import warnings
import time

# Filter out the FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def check_performance_dbscan(df: pd.DataFrame) -> float:
    """
    Uses performance metrics to check the performance or clustering algorithms.

    :param df: dataframe containing the actual and predicted clusters

    :return: the adjusted rand score
    """
    # Similarity between two clustering algorithms
    ari = adjusted_rand_score(df["person"], df["cluster"])

    return ari


def main_test_embeddings_dbscan(dataset: str = None) -> None:
    """
    Main runner for testing different embeddings with
    different parameters for clustering using the DBSCAN algorithm.

    :param dataset: name of the dataset
    """
    # Get all parameters to use
    ebsilons = [i*0.1 for i in range(1, 101)]

    # Iterables to create df with
    models = ["ArcFace", "Dlib", "Facenet", "face_recognition", "VGG_Face"]
    dbscan_params = ["eps"]
    metric = ["ARI", "rel"]

    # Create multi-indexed columns for dataframe
    multi_cols = pd.MultiIndex.from_product([models, metric])
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
            if dataset:
                data = collect_data(f"{dataset}_{model}")
            else:
                data = collect_data(model)

            # Cluster based on the embeddings with the set parameters
            labels = DBSCAN(eps=ebs, min_samples=1, n_jobs=-1, metric="euclidean").fit_predict(data["embedding_tsne"].tolist())

            # Add the clusters to the data
            data1 = data.copy()
            data1["cluster"] = labels

            # Collect the performance metrics
            ari = check_performance_dbscan(data1.copy())

            # Append the performance metrics to the result
            res.append(ari)
            res.append(len(np.unique(labels)) - 17)

        # Add the resulting row to the final dataframe
        result.loc[len(result)] = [round(i, 3) for i in res]

        # Update the progressbar
        pbar.update(1)

    # Save the result
    result.to_excel(f"{f'{dataset}_' if dataset else ''}results_dbscan_parameters.xlsx")

    # Transform columns to strings for latex table
    for col in result.columns:
        if col[1] == "rel":
            result.loc[:, col] = result[col].astype(int).astype(str)
        elif col[1] == "ARI":
            result.loc[:, col] = result[col].apply(lambda x: f"{x:.3f}")
        else:
            result.loc[:, col] = result[col].apply(lambda x: f"{x:.1f}")

    # transform dataframe to latex
    txt = result.to_latex(index=False, multirow=True,
        multicolumn=True,
        escape=False,
        bold_rows=True,
        column_format="c|ll|ll|ll|ll|ll",
    )

    # Write latex table to file
    with open(f"{f'{dataset}_' if dataset else ''}latex_results_dbscan_parameters.txt", "w") as f:
        f.write(txt)


if __name__ == "__main__":
    # Test DBSCAN with different parameters for both the CFI and Embedding on the Wall dataset
    main_test_embeddings_dbscan()
    main_test_embeddings_dbscan("Wies")


    # Check for runtime using the best model
    # Load CFI embeddings
    data = collect_data("face_recognition")
    embeddings = np.array(data["embedding_tsne"].tolist())

    # calculate time needed
    start = time.time()
    DBSCAN(eps=5.4, min_samples=1, n_jobs=-1, metric="euclidean").fit_predict(embeddings)
    end = time.time()

    print(f"Latency for CFI dataset: {end - start}")  # = 0.060s

    # Load Embedding on the Wall embeddings
    data = collect_data("Wies_face_recognition")
    embeddings = np.array(data["embedding_tsne"].tolist())

    # Calculate time needed
    start = time.time()
    DBSCAN(eps=2.3, min_samples=1, n_jobs=-1, metric="euclidean").fit_predict(embeddings)
    end = time.time()

    print(f"Latency for Embedding on the Wall dataset: {end - start}")  # = 0.022s
