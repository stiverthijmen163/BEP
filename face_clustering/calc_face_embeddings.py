import cv2
import numpy as np
import os
from typing import List
from deepface import DeepFace
from tqdm import tqdm
import face_recognition
from ultralytics import YOLO


# Save embeddings and image paths
def save_embeddings(embeddings, image_paths, filename="embeddings.npz"):
    np.savez_compressed(filename, embeddings=embeddings, image_paths=image_paths)
    print(f"Embeddings saved to {filename}")

# Load embeddings and image paths
def load_embeddings(filename="embeddings.npz"):
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
    # Create empty list of embeddings
    embeddings = []

    # Calculate embedding for each image
    for image_path in tqdm(face_paths, desc="extracting face embeddings", ncols=100):
        # Read the image
        # image = face_recognition.load_image_file(image_path)
        image = cv2.imread(image_path)

        model = YOLO("yolo_v12s/yolov12s-face/weights/epoch60.pt")
        result = model(image, verbose=False)

        # Embed the image
        encodings = face_recognition.face_encodings(image, num_jitters=2, model="large")

        # Append embedding to the result
        if encodings:
            embeddings.append(encodings[0])

    save_embeddings(embeddings, face_paths, filename=f"{save_folder}/face_recognition_embeddings.npz")


def main_calc_face_embeddings() -> None:
    """

    """
    # Folder to find all images in
    p = "../data/celebrity-face-image-dataset"

    # Find all image paths in this folder
    image_paths = [
        # f"{root}/{file}"
        os.path.join(root, file)
        for root, _, files in os.walk(p)
        for file in files
        if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    # Generate the face embeddings for all images using different models
    generate_face_embeddings(image_paths, "face_embeddings")


if __name__ == "__main__":
    main_calc_face_embeddings()