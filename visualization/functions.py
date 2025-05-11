import pandas as pd
from ultralytics import YOLO
from PIL import Image
from typing import List, Tuple
import cv2
import numpy as np


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


def detect_faces(img: cv2.Mat) -> Tuple[cv2.Mat, List[List[int]]]:
    """

    """
    # Load in the model to detect faces with
    model = YOLO("../face_detection/yolo_v12s/yolov12s-face/weights/epoch60.pt")
    # print(img.shape)
    # print(img)
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # print(img.shape)
    # Collect all detected faces
    result = model(img, verbose=False)
    # print(result)

    # Collect all bounding boxes
    predictions = []
    for pred in result:
        boxes = pred.boxes
        for box in boxes:
            bb = box.xywhn
            predictions.append(bb[0])

    # Convert prediction to haar-cascade format
    predictions = yolo_to_haar(img, predictions)

    img0 = plot_faces_on_img(img, predictions)

    return img0, predictions


def plot_faces_on_img(img: cv2.Mat, predictions: List[List[int]]) -> cv2.Mat:
    """

    """
    # Plot predicted bounding boxes on image
    for box in predictions:
        x, y, w, h = box
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return img


def plot_faces_on_img_opacity(img: cv2.Mat, df: pd.DataFrame, show_nrs: bool = False) -> cv2.Mat:
    """

    """
    predictions = df["face"].to_list()
    not_opacity = df["use"].to_list()
    nrs = df["nr"].to_list()

    # Plot predicted bounding boxes on image
    for box, n_p, nr in zip(predictions, not_opacity, nrs):
        # print(f"BOX: {box}")
        x, y, w, h = box
        if n_p:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if show_nrs:
                # Draw face number
                cv2.putText(img, f"{nr}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 0), 2, cv2.LINE_AA)
        else:
            overlay = img.copy()
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
            mask = np.zeros_like(img)
            cv2.rectangle(mask, (x, y), (x + w, y + h), (0, 255, 0), 2)
            mask = mask.astype(bool)
            img[mask] = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)[mask]

            if show_nrs:
                overlay = img.copy()
                cv2.putText(
                    overlay, str(nr), (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2, cv2.LINE_AA
                )
                # Blend overlay with the original image to get 50% opacity
                alpha = 0.5
                cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        # if show_nrs:
        #     # Draw face number
        #     cv2.putText(img, f"{nr}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
        #                 (0, 255, 0), 2, cv2.LINE_AA)

    return img

def sort_items(lst: List[str]):
    # Convert items to integers if possible
    lst_int = []
    lst_str = []
    for i in lst:
        try:  # = int
            int(i)
            lst_int.append(int(i))
        except ValueError:  # = string
            lst_str.append(i)

    # Sort the list and convert all items back to strings
    sorted_lst_int = [str(i) for i in sorted(lst_int)]
    sorted_lst_str = sorted(lst_str)

    print(sorted_lst_int)
    print(sorted_lst_str)
    print(sorted_lst_int + sorted_lst_str)

    return sorted_lst_int + sorted_lst_str
