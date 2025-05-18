import sqlite3
import pandas as pd
from ultralytics import YOLO
from typing import List, Tuple
import cv2
import numpy as np
import json
import os


def yolo_to_haar(img: cv2.Mat, boxes: List[List[float]]) -> List[List[int]]:
    """
    Converts YOLO bounding boxes to haar-cascade formatted bounding boxes.

    :param img: image to which the bounding boxes correspond
    :param boxes: list of bounding boxes to convert (format: x_center, y_center, width, height in range 0-1)

    :return: list of bounding boxes in haar-cascade format
    """
    # Initialize list of haar-cascade formatted bounding boxes
    res = []

    # Collect height and width from image
    frame_height, frame_width = img.shape[:2]

    # Convert every bounding box
    for box in boxes:
        x_c, y_x, w, h = box

        # Convert bounding box
        x = int((x_c - 0.5*w) * frame_width)
        y = int((y_x - 0.5*h) * frame_height)
        width = int(w * frame_width)
        height = int(h * frame_height)

        # Save bounding box
        res.append([x, y, width, height])

    return res


def detect_faces(img: cv2.Mat, model: YOLO) -> Tuple[cv2.Mat, List[List[int]]]:
    """
    Detects faces on an image and plots the faces on the image.

    :param img: image to detect faces on
    :param model: model to detect faces with

    :return: image with the faces plotted on top of it, bounding boxes of detected faces
    """
    # Collect all detected faces
    result = model(img, verbose=False)

    # Collect all bounding boxes
    predictions = []
    for pred in result:
        boxes = pred.boxes
        for box in boxes:
            bb = box.xywhn
            predictions.append(bb[0])

    # Convert prediction to haar-cascade format
    predictions = yolo_to_haar(img, predictions)

    # Plot the faces on top of the image
    img0 = plot_faces_on_img(img, predictions)

    return img0, predictions


def plot_faces_on_img(img: cv2.Mat, predictions: List[List[int]]) -> cv2.Mat:
    """
    Plots all faces on an image.

    :param img: image to plot faces on
    :param predictions: list of bounding boxes of detected faces

    :return: image with the faces plotted on top of it
    """
    # Plot predicted bounding boxes on image
    for box in predictions:
        x, y, w, h = box
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return img


def plot_faces_on_img_opacity(img: cv2.Mat, df: pd.DataFrame, show_nrs: bool = False) -> cv2.Mat:
    """
    Plots all faces on an image, making unselected faces less visible and has the option to show the face number.

    :param img: image to plot faces on
    :param df: dataframe containing the bounding boxes, whether the face is selected or not and the face number
    :param show_nrs: whether to show the face number

    :return: image with the faces plotted on top of it
    """
    # Collect all bounding boxes, whether the face is selected or not and the face numbers
    predictions = df["face"].to_list()
    not_opacity = df["use"].to_list()
    nrs = df["nr"].to_list()

    # Plot predicted bounding boxes on the image
    for box, n_p, nr in zip(predictions, not_opacity, nrs):
        x, y, w, h = box
        if n_p:  # Selected face
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if show_nrs:  # Show face number
                cv2.putText(img, f"{nr}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 0), 2, cv2.LINE_AA)
        else:  # Unselected face
            # Plot the face
            overlay = img.copy()
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Create a mask containing the face
            mask = np.zeros_like(img)
            cv2.rectangle(mask, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Add opacity to mask and plot it on the overlay
            mask = mask.astype(bool)
            img[mask] = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)[mask]

            # Show the face number
            if show_nrs:
                # Plot the number on copy
                overlay = img.copy()
                cv2.putText(overlay, f"{nr}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 0), 2, cv2.LINE_AA)

                # Blend overlay with the original image to get 50% opacity
                cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

    return img

def sort_items(lst: List[str]) -> List[str]:
    """
    Sorts the items in a list containing both strings and integers as strings.

    :param lst: list of strings to sort

    :return: sorted list of strings
    """
    # Convert items to integers if possible
    lst_int = []
    lst_str = []
    for i in lst:
        try:  # = int
            lst_int.append(int(i))
        except ValueError:  # = string
            lst_str.append(i)

    # Sort the lists and convert all items back to strings if needed
    sorted_lst_int = [str(i) for i in sorted(lst_int)]
    sorted_lst_str = sorted(lst_str)

    # Return sorted list (numbers before letters)
    return sorted_lst_int + sorted_lst_str


def save_to_db(df_main: pd.DataFrame, df_face: pd.DataFrame, name: str) -> str:
    """
    Saves the two created dataframes to a database.

    :param df_main: dataframe containing all images and extra information
    :param df_face: dataframe containing all faces
    :param name: name of the database

    :return: database name at which the data is actually stored
    """
    # Create path if needed
    if not os.path.exists("databases"):
        os.mkdir("databases")

    # If the inputted name is not set, use the standard format 'database_<number>.db'
    if name == "None" or name == "":
        # Collect all saved databases
        dbs = os.listdir("databases")

        # Check if there are any databases in the standard format
        lst_int = []
        for db in dbs:
            if db.endswith(".db") and "database_" in db:
                try:
                    db = db.strip(".db")
                    if len(db.split("_")) == 2:  # Matches pattern
                        i = int(db.split("_")[1])
                        lst_int.append(i)
                except ValueError:  # Does not match pattern, continue
                    pass

        # Find the number to use in the standard format
        if len(lst_int) > 0:  # Standard format exists, find max + 1
            count = max(lst_int) + 1
        else:  # Standard format does not exist yet
            count = 0

        # Set the name in standard format
        name = f"database_{count}"

    print(f"Saving data to 'databases/{name}.db'")

    # Connect to the database
    conn = sqlite3.connect(f"databases/{name}.db")

    # Change format of main dataframe to match SQL-database requirements
    df_main["img"] = df_main["img"].apply(lambda x: json.dumps(x.tolist()))

    # Save main dataframe to database
    df_main.to_sql("main", conn, if_exists="replace")

    # Change format of the faces dataframe to match SQL-database requirements
    df_face["face"] = df_face["face"].apply(json.dumps)
    df_face["img"] = df_face["img"].apply(lambda x: json.dumps(x.tolist()))
    df_face["embedding"] = df_face["embedding"].apply(lambda x: ",".join(map(str, x.tolist())))
    df_face["embedding_tsne"] = df_face["embedding_tsne"].apply(lambda x: ",".join(map(str, x.tolist())))

    # Remove not needed columns if necessary
    if "level_0" in df_face.columns:
        df_face = df_face.drop("level_0", axis=1)
    if "index" in df_face.columns:
        df_face = df_face.drop("index", axis=1)
    if "use" in df_face.columns:
        df_face = df_face.drop("use", axis=1)

    # Save the faces dataframe to database
    df_face.to_sql("faces", conn, if_exists="replace")

    # Return name of database
    return f"{name}.db"

