from collections import Counter
import numpy as np
import face_recognition
from chinese_whispers import chinese_whispers
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
import os
import networkx as nx
from tqdm import tqdm
import json
from explore_embeddings import collect_data
from dbscan import check_performance_dbscan
import pandas as pd
import openpyxl  # Needed for writing xlsx files
import time
from typing import List


def build_similarity_graph(embeddings: np.array(List[int]), threshold=0.5) -> nx.Graph:
    """
    Create a network graph for the Chinese Whispers clustering algorithm.

    :param embeddings: list of embeddings to use as nodes
    :param threshold: threshold for similarity between two nodes to create an edge with weight=similarity

    :return: graph with all embeddings as nodes and edges corresponding to the similarity
    """
    # Initialize the graph
    G = nx.Graph()

    # Get number of embedding
    num_embeddings = len(embeddings)

    # Add all nodes to the graph
    for i in range(num_embeddings):
        G.add_node(i)

    # Initialize the set of Euclidean distances
    distances = []

    # Calculate distance between all pairs of nodes
    for i in range(num_embeddings):
        for j in range(i + 1, num_embeddings):
            distances.append(np.linalg.norm(embeddings[i] - embeddings[j]))

    # Calculate the maximum distance
    maximum = max(distances)

    # Add edges with similarity weights
    for i in range(num_embeddings):
        for j in range(i + 1, num_embeddings):
            # Calculate normalized similarity
            similarity = 1 - np.linalg.norm(embeddings[i] - embeddings[j])/maximum

            # If similarity meets threshold, add an edge of weight=similarity
            if similarity >= threshold:
                G.add_edge(i, j, weight=similarity)

    return G


def main_test_embeddings_cw(dataset: str = None) -> None:
    """
    Main runner for testing different embeddings with different parameters
    for clustering using the Chinese Whispers algorithm.

    :param dataset: name of the dataset
    """
    # Iterables to create df with
    models = ["ArcFace", "Dlib", "Facenet", "face_recognition", "VGG_Face"]
    metrics = ["ARI", "rel"]
    params = ["thresh"]

    # Create multi-indexed columns for dataframe
    multi_cols = pd.MultiIndex.from_product([models, metrics])
    all_columns = pd.MultiIndex.from_tuples([("params", p) for p in params] + list(multi_cols))
    result = pd.DataFrame(columns=all_columns)

    # Update models to use if dataset is set
    if dataset is not None:
        models = [f"{dataset}_{m}" for m in models]

    # Set all parameters to test
    parameters_val = [round(0.7 + i*0.01, 2) for i in range(30)]

    # Initialize progressbar
    pbar = tqdm(total=len(parameters_val) * len(models), ncols=100, desc="Checking all possible parameters")

    # Iterate over all parameters
    for val in parameters_val:
        # append parameter to the result
        res = [val]

        # cluster using different embedding models
        for m in models:
            # Collect the embeddings
            data = collect_data(m)
            embeddings = np.array(data["embedding_tsne"].tolist())

            # Create graph using the embeddings
            G = build_similarity_graph(embeddings, threshold=val)

            # Use Chinese Whispers to cluster the graph
            G0 = chinese_whispers(G, iterations=100, label_key="label", weighting="top")

            # Add the clusters to the data
            labels = [G0.nodes[i]["label"] for i in range(len(G0.nodes))]
            data["cluster"] = labels

            # Collect the performance metrics
            ari = check_performance_dbscan(data.copy())

            # Append the performance metrics to the result
            res.append(ari)
            res.append(len(np.unique(labels)) - 17)

            # Update the progressbar
            pbar.update(1)

        # Add the resulting row to the final dataframe
        result.loc[len(result)] = [round(i, 3) for i in res]

    # Save the result
    result.to_excel(f"{f'{dataset}_' if dataset else ''}results_CW_parameters.xlsx")

    # Transform columns to strings for latex table
    for col in result.columns:
        if col[1] == "rel":
            result.loc[:, col] = result[col].astype(int).astype(str)
        elif col[1] == "ARI":
            result.loc[:, col] = result[col].apply(lambda x: f"{x:.3f}")
        else:
            result.loc[:, col] = result[col].apply(lambda x: f"{x:.2f}")

    # transform dataframe to latex
    txt = result.to_latex(index=False, multirow=True,
        multicolumn=True,
        escape=False,
        bold_rows=True,
        column_format="c|ll|ll|ll|ll|ll",
    )

    # Write latex table to file
    with open(f"{f'{dataset}_' if dataset else ''}latex_results_CW_parameters.txt", "w") as f:
        f.write(txt)


if __name__ == "__main__":
    # Test Chinese Whispers with different parameters for both the CFI and Embedding on the Wall dataset
    main_test_embeddings_cw()
    main_test_embeddings_cw("Wies")


    # Check for runtime using the best model
    # Load CFI embeddings
    data = collect_data("face_recognition")
    embeddings = np.array(data["embedding_tsne"].tolist())

    # calculate time needed
    start = time.time()
    G = build_similarity_graph(embeddings, threshold=0.94)
    G0 = chinese_whispers(G, iterations=100, label_key="label", weighting="top")
    end = time.time()

    print(f"Latency for CFI dataset: {end - start}")  # = 9.515s

    # Load Embedding on the Wall embeddings
    data = collect_data("Wies_face_recognition")
    embeddings = np.array(data["embedding_tsne"].tolist())

    # calculate time needed
    start = time.time()
    G = build_similarity_graph(embeddings, threshold=0.94)
    G0 = chinese_whispers(G, iterations=100, label_key="label", weighting="top")
    end = time.time()

    print(f"Latency for Embedding on the Wall dataset: {end - start}")  # = 0.626s
