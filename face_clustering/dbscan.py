from collections import Counter
import os
import numpy as np
from sklearn import metrics
from sklearn.cluster import DBSCAN
from chinese_whispers_model import load_embeddings
from sklearn.manifold import TSNE

for embed in os.listdir("face_embeddings"):
    path = os.path.join("face_embeddings", embed)
    print(embed)

    embeddings = load_embeddings(path)
    # print(X)
    # print(X[0])
    # print(X[1])
    # print(embeddings)

    X = embeddings[0]
    image_paths = embeddings[1]

    tsne = TSNE(n_components=2)
    X_t = tsne.fit_transform(X)
    # print(X_t)

    labels = DBSCAN(eps=5, min_samples=4, n_jobs=-1, metric="euclidean").fit_predict(X_t)
    # print(Counter(db))
    # labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)
    print(labels)
    counter = Counter(labels)
    print(counter, "\n")
