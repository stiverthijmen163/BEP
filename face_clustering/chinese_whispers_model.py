import numpy as np
import face_recognition
from chinese_whispers import chinese_whispers
from scipy.spatial.distance import cosine
import os
import networkx as nx
from tqdm import tqdm
import json


def extract_face_embeddings(image_paths):
    """Extract face embeddings from images using face_recognition."""
    embeddings = []
    for image_path in tqdm(image_paths, desc="extracting face embeddings", ncols=100):
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image, num_jitters=2, model="large")
        if encodings:
            embeddings.append(encodings[0])
    return embeddings


def build_similarity_graph(embeddings, threshold=0.5):
    """Create a NetworkX graph for Chinese Whispers clustering."""
    G = nx.Graph()
    num_embeddings = len(embeddings)

    # Add nodes
    for i in range(num_embeddings):
        G.add_node(i)

    # Add edges with similarity weights
    for i in range(num_embeddings):
        for j in range(i + 1, num_embeddings):
            similarity = 1 - cosine(embeddings[i], embeddings[j])  # Cosine similarity<< LOOK AT DIMENSION >>
            # print(similarity)
            if similarity >= threshold:
                G.add_edge(i, j, weight=similarity)

    return G


def save_clusters(clusters, filename="clusters.json"):
    with open(filename, "w") as f:
        json.dump(clusters, f, indent=4)


def load_clusters(filename="clusters.json"):
    with open(filename, "r") as f:
        return json.load(f)

# Save embeddings and image paths
def save_embeddings(embeddings, image_paths, filename="embeddings.npz"):
    np.savez_compressed(filename, embeddings=embeddings, image_paths=image_paths)
    print(f"Embeddings saved to {filename}")

# Load embeddings and image paths
def load_embeddings(filename="embeddings.npz"):
    data = np.load(filename, allow_pickle=True)
    return data["embeddings"], data["image_paths"]

def normalize_embeddings(embeddings):
    return np.array([e / np.linalg.norm(e) for e in embeddings])


if __name__ == "__main__":
    # # Example usage:
    # # image_paths = ["face1.jpg", "face2.jpg", "face3.jpg"]  # Replace with actual paths
    # p = "../data/celebrity-face-image-dataset/1/Celebrity Faces Dataset"
    # # Collect all image paths
    # image_paths = [
    #     # f"{root}/{file}"
    #     os.path.join(root, file)
    #     for root, _, files in os.walk(p)
    #     for file in files
    #     if file.lower().endswith((".jpg", ".jpeg", ".png"))
    # ]
    # # image_paths = [f"{p}/{i}" for i in os.listdir(p)]
    # # print(image_paths)

    # embeddings = extract_face_embeddings(image_paths)
    # save_embeddings(embeddings, image_paths)
    embeddings, image_paths = load_embeddings()
    # embeddings = normalize_embeddings(embeddings)
    # print(embeddings)
    print(len(embeddings))
    print("Creating graph...")
    G = build_similarity_graph(embeddings)

    # Apply Chinese Whispers clustering
    print("Applying Chinese Whispers...")
    labels = chinese_whispers(G, iterations=100, label_key="label", weighting="top")
    print("Done!")

    print(labels)
    for i in labels:
        print(i)
    # Group results by cluster
    clusters = {}
    for idx, label in enumerate(labels):
        # print(idx, label)
        clusters.setdefault(label, []).append(image_paths[idx])

    # print("Clusters:", clusters)
    print("Nr of clusters: ", len(clusters))

    save_clusters(clusters)
