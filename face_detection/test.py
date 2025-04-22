import cv2
import insightface.app.common
import numpy as np
import pandas as pd
import pyarrow  # Needed for parquet files
import fastparquet  # Needed for parquet files
import os

from numpy import floating
from tqdm import tqdm
from base_model import detect_faces_haar_cascade_recognition
from typing import List, Tuple, Any
from ultralytics import YOLO
import matplotlib.pyplot as plt
import openpyxl  # Needed for writing xlsx files
from insightface.app import FaceAnalysis
import time
from sklearn import metrics
import seaborn as sns


def show_masks(img: cv2.Mat, expected_masks: cv2.Mat, predicted_boxes: List[List[int]], save_path: str) -> None:
    """
    Creates and saves an images with the expected and predicted bounding boxes.

    :param img: image to plot on
    :param expected_masks: the expected bounding boxes
    :param predicted_boxes: the predicted bounding boxes
    :param save_path: path to save the image to
    """
    # Convert the expected mask to a colored scale
    mask_colored = expected_masks * 255

    # Blend expected mask with image
    overlay = cv2.addWeighted(img, 0.7, mask_colored, 0.3, 0)

    # Plot predicted bounding boxes on image
    for box in predicted_boxes:
        x, y, w, h = box
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Create path to save file if needed
    if not os.path.exists("/".join(save_path.split("/")[:-1])):
        os.makedirs("/".join(save_path.split("/")[:-1]))

    # Save the image
    cv2.imwrite(save_path, overlay)


def save_cutout_faces(img: cv2.Mat, predicted_boxes: List[List[int]], save_path: str) -> None:
    """
    Saves all faces as new images.

    :param img: image to take faces from
    :param predicted_boxes: the predicted bounding boxes
    :param save_path: path to save the cut-out faces to
    """
    # Create path to save file if needed
    if not os.path.exists("/".join(save_path.split("/")[:-1])):
        os.makedirs("/".join(save_path.split("/")[:-1]))

    # Initialize face count
    count = 0

    for box in predicted_boxes:
        # Cut out a face
        x, y, w, h = box
        face_img = img[y:y+h, x:x+w]

        # Save the face
        cv2.imwrite(f"{save_path[:-4]}_{count}.jpg", face_img)

        count += 1


def create_conf_matrix(actual: List[str], predicted: List[str], save_folder: str, model_name: str,
                       threshold: float) -> None:
    """
    Creates a confidence matrix based on a list of predictions and te actual values (either 'face' or 'background').

    :param actual: list of expected values
    :param predicted: list of predicted values
    :param save_folder: path of folder to save the image to
    :param model_name: name of the model
    :param threshold: threshold for IoU used for calculation of tp, tf, np and np
    """
    # Create confusion matrix
    confusion_matrix = metrics.confusion_matrix(actual, predicted, labels=["face", "background"])

    # Create plot for confusion matrix
    plt.figure(figsize=(7, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt="g", cmap="Blues", xticklabels=["face", "background"],
                yticklabels=["face", "background"])
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    plt.title(f"{model_name} with IoU ≥ {threshold}", fontsize=14)

    # Create folder to save plot to if needed
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Save the confusion matrix
    plt.savefig(f"{save_folder}/{model_name}_thres_{''.join(str(threshold).split('.'))}.jpg")


def iou(img: cv2.Mat, boxes_det: List[List[int]], boxes_exp: List[List[int]], save_path: str = None) -> float:
    """
    Calculates the Intersection over Union (IoU) over all predicted bounding boxes in an image.

    :param img: image on which the predictions are predicted
    :param boxes_det: list of predicted bounding boxes
    :param boxes_exp: list of expected bounding boxes
    :param save_path: path to save all resulting images to, set to None to not plot bounding boxes on images

    :return: IoU over all predicted bounding boxes
    """
    # Initialize masks for predicted and expected bounding boxes respectively
    mask0 = np.zeros_like(img)
    mask1 = np.zeros_like(img)

    # Set every pixel inside a predicted bounding box to 1
    for x, y, w, h in boxes_det:
        mask0[y:y+h, x:x+w] = 1

    # Set every pixel inside an expected bounding box to 1
    for x, y, w, h in boxes_exp:
        mask1[y:y+h, x:x+w] = 1

    # Calculate the intersection and union of both masks
    intersection = np.logical_and(mask0, mask1).sum()
    union = np.logical_or(mask0, mask1).sum()

    # Plot bounding boxes on image
    if save_path is not None:
        show_masks(img, mask1, boxes_det, save_path)

    # Return IoU
    return intersection / union


def calc_accuracy(img: cv2.Mat, boxes_det: List[List[int]], boxes_exp: List[List[int]],
                  iou_threshold: float) -> Tuple[float, List[str], List[str]]:
    """
    Calculates the accuracy of the predicted bounding boxes in an image,
    with true positive having an IoU with the actual bounding box over a certain threshold.

    :param img: image on which the predictions are predicted
    :param boxes_det: list of predicted bounding boxes
    :param boxes_exp: list of expected bounding boxes
    :param iou_threshold: threshold for which the IoU must be over to have a true positive

    :return: the accuracy, list of actual values and a list of predicted values ('face' or 'background')
    """
    # Initialize true positives, false positives and false negatives
    tp = 0
    fp = 0
    fn = 0

    # Initialize list of actual and predicted values
    actual = []
    predicted = []

    # Look at every predicted box
    for box in boxes_det:
        # Initialize list of IoU-scores and set the maximum IoU so far
        iou_scores = []
        max_iou = -1

        # Iterate over all expected boxes and calculate the IoU
        for bb in boxes_exp:
            iou_scores.append(iou(img, [box], [bb]))

        # Collect the maximum IoU
        if len(iou_scores) != 0:
            max_iou = max(iou_scores)

        # If max IoU is over the threshold, prediction is true positive
        if max_iou >= iou_threshold:
            # Update variables
            tp += 1
            actual.append("face")
            predicted.append("face")

            # Remove the used actual face from the list to prevent it from being used twice
            boxes_exp.pop(iou_scores.index(max_iou))
        else:  # False positive otherwise
            fp += 1
            actual.append("background")
            predicted.append("face")

    # All remaining expected faces are not predicted by the model, thus false negative
    fn += len(boxes_exp)
    actual += ["face"] * len(boxes_exp)
    predicted += ["background"] * len(boxes_exp)

    return tp / (tp + fp + fn), actual, predicted


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


def insightface_to_haar(boxes: List[insightface.app.common.Face]) -> List[List[int]]:
    """
    Converts bounding boxes from insightface to haar-cascade formatted bounding boxes.

    :param boxes: list of bounding boxes in insightface format

    :return: list of bounding boxes in haar-cascade format
    """
    # Initialize list of haar-cascade formatted bounding boxes
    res = []

    # Convert every bounding box
    for face in boxes:
        box = face.bbox.astype(int)  # Set values to integers
        x_l, y_t, x_r, y_b = box

        # Convert bounding box
        w = x_r - x_l
        h = y_b - y_t

        # Save bounding box
        res.append([x_l, y_t, w, h])

    return res


def collect_yolo_ann(img_path: str) -> List[List[float]]:
    """
    Collects all annotations in YOLO format.

    :param img_path: path to image

    :return: list of bounding boxes in YOLO format
    """
    # Convert image path to label path
    lbl_path = f"{img_path[:-4].replace('images', 'labels')}.txt"

    # Open annotation file
    with open(lbl_path, "r") as f:
        lines = f.readlines()  # Read the lines
        expected_boxes = []  # Initialize list of bounding boxes

        # Iterate over every line
        for line in lines:
            box = line.strip()[2:].split()  # Remove class from annotation
            expected_boxes.append([float(i) for i in box])  # Save bounding box

    return expected_boxes


def calc_iou_acc_latency(p: str, img0: cv2.Mat, iou_s: List[float], acc_s_05: List[float], acc_s_07: List[float],
                         actu_05: List[str], actu_07: List[str], predi_05: List[str], predi_07: List[str],
                         rt: List[float], s: float, e: float, predicted, save_path: str)\
        -> Tuple[List[float], List[float], List[str], List[str], List[str], List[str], List[float]]:
    """
    Calculates the IoU, Accuracy at different IoU threshold, latency
    and stores the predicted and actual values ('face' or 'background').

    :param p: path to the image
    :param img0: image on which the predictions are predicted
    :param iou_s: list of IoU scores
    :param acc_s_05: list of accuracy scores at IoU threshold == 0.5
    :param acc_s_07: list of accuracy scores at IoU threshold == 0.7
    :param actu_05: list of actual values at IoU threshold == 0.5 ('face' or 'background')
    :param actu_07: list of actual values at IoU threshold == 0.7 ('face' or 'background')
    :param predi_05: list of predicted values at IoU threshold == 0.5 ('face' or 'background')
    :param predi_07: list of predicted values at IoU threshold == 0.7 ('face' or 'background')
    :param rt: list of latency values
    :param s: start of model runtime
    :param e: end of model runtime
    :param predicted: list of predicted bounding boxes
    :param save_path: path to save plotted bounding boxes to

    :return: all updated lists of IoU-scores, accuracy-scores, latency values and
    """
    # Collect expected annotations and convert them to haar-cascade format
    expected_yolo = collect_yolo_ann(p)
    expected = yolo_to_haar(img0, expected_yolo)

    # Calculate the IoU
    iou_s.append(iou(img0, predicted, expected, save_path))

    # Calculate the accuracy at different IoU thresholds
    acc_05, act_05, pred_05 = calc_accuracy(img0, predicted.copy(), expected.copy(), 0.5)
    acc_07, act_07, pred_07 = calc_accuracy(img0, predicted.copy(), expected.copy(), 0.7)

    # Append the results to their corresponding list
    acc_s_05.append(acc_05)
    acc_s_07.append(acc_07)
    actu_05 = actu_05 + act_05
    predi_05 = predi_05 + pred_05
    actu_07 = actu_07 + act_07
    predi_07 = predi_07 + pred_07
    rt.append(e - s)

    return acc_s_05, acc_s_07, actu_05, predi_05, actu_07, predi_07, rt


def test_haar_fr(test_data_path: str, model_name: str) -> List[str | floating]:
    """
    Tests a haar-cascade model combined with the face-recognition package.

    :param test_data_path: path to test data
    :param model_name: Name of the model

    :return: list of name of the model, average IoU-score, accuracy scores and latency
    """
    # Collect all images
    test_images = os.listdir(test_data_path)

    # Initialize variables
    iou_scores = []
    accuracy_scores_05 = []
    accuracy_scores_07 = []
    runtime = []
    actual_05 = []
    predicted_05 = []
    actual_07 = []
    predicted_07 = []

    # Iterate over all images
    for test_img in tqdm(test_images, desc="Testing haar cascade + face recognition model", ncols=100):
        # Set image path and open image
        path = f"{test_data_path}/{test_img}"
        img = cv2.imread(path)

        # Run model on the image and measure the time taken
        start = time.time()
        predictions = detect_faces_haar_cascade_recognition(img)
        end = time.time()

        # Update all variables by calculating IoU, Accuracy scores and Latency
        accuracy_scores_05, accuracy_scores_07, actual_05, predicted_05, actual_07, predicted_07, runtime = (
            calc_iou_acc_latency(path, img, iou_scores, accuracy_scores_05, accuracy_scores_07, actual_05, actual_07,
                                 predicted_05, predicted_07, runtime, start, end, predictions,
                                 f"data/{model_name}/full_images/{test_img}"))

    # Create confidence matrices at different IoU thresholds
    create_conf_matrix(actual_05, predicted_05, f"plots", "haar cascade + dlib", 0.5)
    create_conf_matrix(actual_07, predicted_07, f"plots", "haar cascade + dlib", 0.7)

    return [model_name, np.average(iou_scores).round(3), np.average(accuracy_scores_05).round(3),
            np.average(accuracy_scores_07).round(3), np.average(runtime).round(3)]


def test_yolo(test_data_path: str, model_path: str, model_name: str) -> list[str | floating]:
    """
    Tests a YOLO model.

    :param test_data_path: path to test data
    :param model_path: path to the model
    :param model_name: name of the model

    :return: list of name of the model, average IoU-score, accuracy scores and latency
    """
    # Collect all images
    test_images = os.listdir(test_data_path)

    # Initialize variables
    iou_scores = []
    accuracy_scores_05 = []
    accuracy_scores_07 = []
    runtime = []
    actual_05 = []
    predicted_05 = []
    actual_07 = []
    predicted_07 = []

    # Load the model
    model = YOLO(model_path)

    # Iterate over all images
    for test_img in tqdm(test_images, desc=f"Testing YOLO model: ({model_name})", ncols=100):
        # Set image path and open image
        path = f"{test_data_path}/{test_img}"
        img = cv2.imread(path)

        # Run model on the image and measure the time taken
        start = time.time()
        result = model(img, verbose=False)
        end = time.time()

        # Collect all bounding boxes
        predictions = []
        for pred in result:
            boxes = pred.boxes
            for box in boxes:
                bb = box.xywhn
                predictions.append(bb[0])

        # Convert prediction to haar-cascade format
        predictions = yolo_to_haar(img, predictions)

        # Update all variables by calculating IoU, Accuracy scores and Latency
        accuracy_scores_05, accuracy_scores_07, actual_05, predicted_05, actual_07, predicted_07, runtime =(
            calc_iou_acc_latency(path, img, iou_scores, accuracy_scores_05, accuracy_scores_07, actual_05, actual_07,
                                 predicted_05, predicted_07, runtime, start, end, predictions,
                                 f"data/{model_name}/full_images/{test_img}"))

    # Create confidence matrices at different IoU thresholds
    create_conf_matrix(actual_05, predicted_05, f"plots", model_name, 0.5)
    create_conf_matrix(actual_07, predicted_07, f"plots", model_name, 0.7)

    return [model_name, np.average(iou_scores).round(3), np.average(accuracy_scores_05).round(3),
            np.average(accuracy_scores_07).round(3), np.average(runtime).round(3)]


def test_insightface(test_data_path: str, model_name: str) -> list[str | floating]:
    """
    Tests InsightFace model.

    :param test_data_path: path to test data
    :param model_name: name of the model

    :return: list of name of the model, average IoU-score, accuracy scores and latency
    """
    # Collect all images
    test_images = os.listdir(test_data_path)

    # Initialize variables
    iou_scores = []
    accuracy_scores_05 = []
    accuracy_scores_07 = []
    runtime = []
    actual_05 = []
    predicted_05 = []
    actual_07 = []
    predicted_07 = []

    # Initialize the model
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], allowed_modules=['detection'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    # Iterate over all images
    for test_img in tqdm(test_images, desc=f"Testing InsightFace", ncols=100):
        # Set image path and open image
        path = f"{test_data_path}/{test_img}"
        img = cv2.imread(path)

        # Run model on the image and measure the time taken
        start = time.time()
        faces = app.get(img)
        end = time.time()

        # Convert prediction to haar-cascade format
        predictions = insightface_to_haar(faces)

        # Update all variables by calculating IoU, Accuracy scores and Latency
        accuracy_scores_05, accuracy_scores_07, actual_05, predicted_05, actual_07, predicted_07, runtime = (
            calc_iou_acc_latency(path, img, iou_scores, accuracy_scores_05, accuracy_scores_07, actual_05, actual_07,
                                 predicted_05, predicted_07, runtime, start, end, predictions,
                                 f"data/{model_name}/full_images/{test_img}"))

    # Create confidence matrices at different IoU thresholds
    create_conf_matrix(actual_05, predicted_05, f"plots", "insightface", 0.5)
    create_conf_matrix(actual_07, predicted_07, f"plots", "insightface", 0.7)

    return [model_name, np.average(iou_scores).round(3), np.average(accuracy_scores_05).round(3),
            np.average(accuracy_scores_07).round(3), np.average(runtime).round(3)]


def main_test_detection() -> None:
    """
    Main runner for testing all models for face detection.
    """
    results = []

    # Set path to the test images
    path = "../data/Face detection.v1i.yolov8/test/images"

    # Test the haar-cascade + face-recognition model
    results.append(test_haar_fr(path, "haar_cascade + dlib (base model)"))

    # Test the InsightFace model
    results.append(test_insightface(path, "insightface"))

    # Test the yolov6n-face model
    model_path = "downloaded_models/yolov6n-face.pt"
    results.append(test_yolo(path, model_path, "yolov6n-face"))

    # Test the yolov11l-face model
    model_path = "downloaded_models/yolov11l-face.pt"
    results.append(test_yolo(path, model_path, "yolov11l-face"))

    # Test the self-trained model
    model_path = "yolo_v12s/yolov12s-face/weights/epoch60.pt"
    results.append(test_yolo(path, model_path, "yolov12s-face"))

    # Save results to excel
    df_res = pd.DataFrame(results,
                          columns=["Model", "IoU (entire image)", "Accuracy (IoU ≥ 0.5)", "Accuracy (IoU ≥ 0.7)",
                                   "Latency (s)"]).set_index("Model")
    df_res.to_excel("results_face_detection.xlsx", index=True)


if __name__ == "__main__":
    main_test_detection()
