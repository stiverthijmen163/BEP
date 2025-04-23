import cv2
import numpy as np
import os
from typing import List, Tuple
from deepface import DeepFace
from tqdm import tqdm
import face_recognition
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import sqlite3


def save_embeddings(embeddings :List, image_paths: List[str], filename: str = "embeddings.npz") -> None:
    """
    Saves embeddings and their respective image paths.

    :param embeddings: list of embeddings to save
    :param image_paths: list of image paths respective to the embeddings
    :param filename: file to save the embeddings to, must be .npz-file
    """
    # Save the embeddings as npz file
    np.savez_compressed(filename, embeddings=embeddings, image_paths=image_paths)
    print(f"Embeddings saved to {filename}")

# Load embeddings and image paths
def load_embeddings(filename: str = "embeddings.npz") -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads embeddings and their respective image paths.

    :param filename: file to load the embeddings from, must be .npz-file

    :return: embeddings and their respective image paths
    """
    # Load the .npz file
    data = np.load(filename, allow_pickle=True)
    return data["embeddings"], data["image_paths"]


def generate_face_embeddings(face_paths: List[str], save_folder: str) -> None:
    """
    Calculates face embeddings using different models and saves them with in the save folder.

    :param face_paths: List of paths to face images.
    :param save_folder: Folder to save face embeddings.
    """
    # Create save directory if needed
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)

    # Set all models to use for calculating embeddings
    models = ["VGG-Face", "Facenet", "OpenFace", "DeepFace", "Dlib", "ArcFace"]

    # Create a new list of embeddings for each model
    for model in models:
        embeddings = []

        # Calculate embedding for each image
        for face_path in tqdm(face_paths, ncols=100, desc=f"Embedding images using {model}"):
            # Read the image
            face = cv2.imread(face_path)

            # Embed the image
            embedding = DeepFace.represent(face, model_name=model, enforce_detection=False)

            # Append embedding to the result
            embeddings.append(embedding[0]["embedding"])

        # Save embeddings
        save_embeddings(embeddings, face_paths, filename=f"{save_folder}/{model}_embeddings.npz")


def extract_face_recognition_embeddings(face_paths: List[str], save_folder: str) -> None:
    """
    Extracts face embeddings from images using face_recognition.

    :param face_paths: List of paths to face images.
    :param save_folder: Folder to save face embeddings.
    """
    # Create save directory if needed
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)

    # Create empty list of embeddings
    embeddings = []

    # Calculate embedding for each image
    for image_path in tqdm(face_paths, desc="extracting face embeddings using face-recognition", ncols=100):
        # Read the image
        image = cv2.imread(image_path)
        height, width = image.shape[:2]

        # Embed the image
        encodings = face_recognition.face_encodings(image, num_jitters=2, model="large",
                                                    known_face_locations=[(0,width,height,0)])

        # Append embedding to the result
        if encodings:
            embeddings.append(encodings[0])

    # Save the embeddings
    save_embeddings(embeddings, face_paths, filename=f"{save_folder}/face_recognition_embeddings.npz")


def main_calc_face_embeddings() -> None:
    """
    Main runner for calculating all face embeddings
    """
    # Folder to find all images in
    p = "data/celebrity-face-image-dataset"

    # Find all image paths in this folder
    image_paths = [
        os.path.join(root, file).replace("\\", "/")
        for root, _, files in os.walk(p)
        for file in files
        if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    # Generate the face embeddings for all images using face-recognition
    extract_face_recognition_embeddings(image_paths, "face_embeddings")

    # Generate the face embeddings for all images using different models
    generate_face_embeddings(image_paths, "face_embeddings")


def main_process_face_embeddings() -> None:
    """
    Saves all embeddings to SQL database and reduces them to 2 dimensions.
    """
    # Create connection to the database
    database_file = "face_embeddings/data.db"
    os.makedirs(os.path.dirname(database_file), exist_ok=True)
    conn = sqlite3.connect(database_file)

    # Load in embeddings for each model
    for p in os.listdir("face_embeddings"):
        if p.endswith(".npz"):
            # Load embeddings
            emb, paths = load_embeddings(f"face_embeddings/{p}")

            # Reduce embeddings by using TSNE
            tsne = TSNE(n_components=2)
            emb_tsne = tsne.fit_transform(emb)

            # Reduce embeddings by using PCA
            pca = PCA(n_components=2)
            emb_pca = pca.fit_transform(emb)

            # Collect the name of the model used for page name
            model = "_".join(p.split("_")[:-1]).replace("-", "_")

            # Create a dataframe containing all embeddings
            res = pd.DataFrame({
                "id_path": paths,
                "embedding": [",".join(map(str, e.tolist())) for e in emb],
                "embedding_pca": [",".join(map(str, e.tolist())) for e in emb_pca],
                "embedding_tsne": [",".join(map(str, e.tolist())) for e in emb_tsne],
            })

            # Save df to SQL database
            res.to_sql(model, conn, if_exists="replace")

    # Collect the person for each path
    persons = [path.split("/")[-2] for path in paths]

    # Create a dataframe containing all persons
    res = pd.DataFrame({
        "id_path": paths,
        "person": persons
    })

    # save df to SQL database
    res.to_sql("persons", conn, if_exists="replace")

if __name__ == "__main__":
    main_calc_face_embeddings()
    main_process_face_embeddings()
