from collections import Counter
import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import DBSCAN
# from chinese_whispers_model import load_embeddings
# from sklearn.manifold import TSNE
from explore_embeddings import collect_data
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import fowlkes_mallows_score
import openpyxl  # Needed for writing xlsx files
from tqdm import tqdm
from joblib import Parallel, delayed


def plot_clusters_from_dbscan(data: pd.DataFrame, save_path: str, name: str) -> None:
    """

    """
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=data,
        x="x",
        y="y",
        hue="person",
        hue_order=sorted(data["person"].unique()),
        palette=sns.color_palette("tab20", n_colors=len(data["cluster"].unique())),
        s=30
    )

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Person")
    plt.title(f"Cluster of Face Embeddings by DBSCAN")
    plt.tight_layout()

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(f"{save_path}/{name}.jpg")


def process_combination(ebs, sample, models):
    res = [ebs, sample]
    for model in models:
        data = collect_data(model)  # Consider caching this if it's slow
        labels = DBSCAN(eps=ebs, min_samples=sample, n_jobs=-1, metric="euclidean").fit_predict(
            data["embedding_tsne"].tolist()
        )
        data["cluster"] = labels
        ari, fmi = check_performance_dbscan(data)
        res.append(ari)
        res.append(fmi)
    return res

def main_test_embeddings_dbscan():
    ebsilons = [i * 0.1 for i in range(1, 101)]
    samples = [1]
    models = ["ArcFace", "Dlib", "Facenet", "face_recognition", "VGG_Face"]
    dbscan_params = ["eps", "min_samples"]
    metrics = ["ARI", "FMI"]
    multi_cols = pd.MultiIndex.from_product([models, metrics])
    all_columns = pd.MultiIndex.from_tuples([("params", p) for p in dbscan_params] + list(multi_cols))

    combinations = [(e, s) for e in ebsilons for s in samples]
    results = Parallel(n_jobs=-1)(delayed(process_combination)(e, s, models) for e, s in tqdm(combinations))

    result_df = pd.DataFrame(results, columns=all_columns)
    result_df = result_df.round(3)
    result_df.to_excel("results_dbscan_parameters.xlsx")


def check_performance_dbscan(df: pd.DataFrame) -> Tuple[float, float]:
    """

    """
    # Similarity between two clustering algorithms
    ari = adjusted_rand_score(df["person"], df["cluster"])
    # print(f"Adjusted Rand Index: {ari:.4f}")

    # sqrt(precision*recall)
    fmi = fowlkes_mallows_score(df["person"], df["cluster"])
    # print(f"Fowlkes Mallows Index: {fmi:.4f}")

    return ari, fmi
#
#
# def main_test_embeddings_dbscan() -> None:
#     """
#
#     """
#     ebsilons = [i*0.1 for i in range(1, 101)]
#     samples = [i for i in range(1, 21)]
#     # print(ebsilons)
#     # print(samples)
#     models = ["ArcFace", "Dlib", "Facenet", "face_recognition", "VGG_Face"]
#     dbscan_params = ["eps", "min_samples"]
#     metrics = ["ARI", "FMI"]
#
#     # Create multi-indexed columns for dataframe
#     multi_cols = pd.MultiIndex.from_product([models, metrics])
#     all_columns = pd.MultiIndex.from_tuples([("params", p) for p in dbscan_params] + list(multi_cols))
#     result = pd.DataFrame(columns=all_columns)
#
#     # print(result)
#     pbar = tqdm(total=len(ebsilons)*len(samples), ncols=100, desc="Checking all possible parameters")
#
#     for ebs in ebsilons:
#         for sample in samples:
#             res = [ebs, sample]
#             for model in models:
#                 # if p.endswith(".npz"):
#                 #     model = "_".join(p.split("_")[:-1]).replace("-", "_")
#                 # print(model)
#
#                 data = collect_data(model)
#
#                 # labels = DBSCAN(eps=5, min_samples=4, n_jobs=-1, metric="euclidean").fit_predict(data["embedding_tsne"].tolist())
#                 # labels = DBSCAN(eps=4, min_samples=1, n_jobs=-1, metric="euclidean").fit_predict(data["embedding_tsne"].tolist())
#                 labels = DBSCAN(eps=ebs, min_samples=sample, n_jobs=-1, metric="euclidean").fit_predict(data["embedding_tsne"].tolist())
#
#                 n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
#                 n_noise_ = list(labels).count(-1)
#
#                 # print("Estimated number of clusters: %d" % n_clusters_)
#                 # print("Estimated number of noise points: %d" % n_noise_)
#                 # print(labels)
#                 # counter = Counter(labels)
#                 # print(counter, "\n")
#
#                 data1 = data.copy()
#                 data1["cluster"] = labels
#                 # data1["x"] = data1["embedding_tsne"].apply(lambda lst: lst[0])
#                 # data1["y"] = data1["embedding_tsne"].apply(lambda lst: lst[1])
#
#                 # print(data1[["person", "cluster"]])
#                 # plot_clusters_from_dbscan(data1.copy(), f"plots/DBSCAN", name=p)
#                 ari, fmi = check_performance_dbscan(data1.copy())
#                 res.append(ari)
#                 res.append(fmi)
#
#             result.loc[len(result)] = [round(i, 3) for i in res]
#             pbar.update(1)
#             # print(result)
#         # break
#     result.to_excel("results_dbscan_parameters.xlsx")




if __name__ == "__main__":
    main_test_embeddings_dbscan()
    # for embed in os.listdir("face_embeddings"):
    #     path = os.path.join("face_embeddings", embed)
    #     print(embed)
    #
    #     embeddings = load_embeddings(path)
    #     # print(X)
    #     # print(X[0])
    #     # print(X[1])
    #     # print(embeddings)
    #
    #     X = embeddings[0]
    #     image_paths = embeddings[1]
    #
    #     tsne = TSNE(n_components=2)
    #     X_t = tsne.fit_transform(X)
    #     # print(X_t)
    #
    #     labels = DBSCAN(eps=5, min_samples=4, n_jobs=-1, metric="euclidean").fit_predict(X_t)
    #     # print(Counter(db))
    #     # labels = db.labels_
    #
    #     # Number of clusters in labels, ignoring noise if present.
    #     n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    #     n_noise_ = list(labels).count(-1)
    #
    #     print("Estimated number of clusters: %d" % n_clusters_)
    #     print("Estimated number of noise points: %d" % n_noise_)
    #     print(labels)
    #     counter = Counter(labels)
    #     print(counter, "\n")
    #     break
