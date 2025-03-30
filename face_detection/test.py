import cv2
import insightface.app.common
import pandas as pd
import pyarrow  # Needed for parquet files
import fastparquet  # Needed for parquet files
import os
import requests
from PIL import Image
import io

from numpy import floating
from tqdm import tqdm
from base_model import *
from typing import List, Tuple, Any
from ultralytics import YOLO
import matplotlib.pyplot as plt
import openpyxl  # Needed for writing xlsx files
from insightface.app import FaceAnalysis


def overlay_mask_cv2(img, mask):
    # Convert mask to 3 channels
    if len(mask.shape) == 2:  # Only convert if mask is single-channel
        mask_colored = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)
    else:
        mask_colored = mask * 255

    # Blend mask with image
    overlay = cv2.addWeighted(img, 0.7, mask_colored, 0.3, 0)

    # Show using OpenCV
    cv2.imshow("Mask Overlay", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_masks(img: cv2.Mat, expected_masks: cv2.Mat, predicted_boxes: List[List[int]], save_path: str) -> None:
    """

    """
    mask_colored = expected_masks * 255

    # Blend mask with image
    overlay = cv2.addWeighted(img, 0.7, mask_colored, 0.3, 0)
    for box in predicted_boxes:
        x, y, w, h = box
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)

    if not os.path.exists("/".join(save_path.split("/")[:-1])):
        os.makedirs("/".join(save_path.split("/")[:-1]))

    cv2.imwrite(save_path, overlay)


def iou(img: cv2.Mat, boxes_det: List[List[int]], boxes_exp: List[List[int]], save_path: str = None) -> float:
    """

    :param img:
    :param boxes_det:
    :param boxes_exp:
    :return:
    """
    frame_height, frame_width = img.shape[:2]

    mask0 = np.zeros_like(img)
    mask1 = np.zeros_like(img)

    # Set pixels inside detected boxes to 1
    for x, y, w, h in boxes_det:
        mask0[y:y+h, x:x+w] = 1

    # Set pixels inside expected boxes to 1
    for x, y, w, h in boxes_exp:
        mask1[y:y+h, x:x+w] = 1

    # tp = 0
    # fp = 0
    # fn = 0
    #
    # for i in range(frame_height):
    #     for j in range(frame_width):
    #         if mask1[i][j][0] == 1 and mask0[i][j][0] == 1:
    #             tp += 1
    #         elif mask1[i][j][0] == 0 and mask0[i][j][0] == 1:
    #             fp += 1
    #         elif mask1[i][j][0] == 1 and mask0[i][j][0] == 0:
    #             fn += 1
    intersection = np.logical_and(mask0, mask1).sum()
    union = np.logical_or(mask0, mask1).sum()

    # print(mask1[263][381])
    # print(frame_width, frame_height)
    # print(mask0)
    #
    # print(boxes_det)
    # print(boxes_exp)
    #
    # overlay_mask_cv2(img, mask0)
    # overlay_mask_cv2(img, mask1)

    if save_path is not None:
        show_masks(img, mask1, boxes_det, save_path)

    # return tp / (tp + fp + fn)
    return intersection / union


def calc_accuracy(img: cv2.Mat, boxes_det: List[List[int]], boxes_exp: List[List[int]], iou_threshold: float) -> float:
    """

    """
    tp = 0
    fp = 0
    fn = 0
    for box in boxes_det:
        iou_scores = []
        max_iou = -1
        for bb in boxes_exp:
            iou_scores.append(iou(img, [box], [bb]))

        if len(iou_scores) != 0:
            max_iou = max(iou_scores)
        if max_iou >= iou_threshold:
            tp += 1
            boxes_exp.pop(iou_scores.index(max_iou))
        else:
            fp += 1
    fn += len(boxes_exp)

    return tp / (tp + fp + fn)



def yolo_to_haar(img: cv2.Mat, boxes: List[List[float]]) -> List[List[int]]:
    """

    :param img:
    :param boxes:
    :return:
    """
    res = []
    frame_height, frame_width = img.shape[:2]
    for box in boxes:
        # img0 = cv2.imread("data/wider_face/val/images/0_Parade_Parade_0_913.jpg")
        x_c, y_x, w, h = box
        x = int((x_c - 0.5*w) * frame_width)
        y = int((y_x - 0.5*h) * frame_height)
        width = int(w * frame_width)
        height = int(h * frame_height)

        res.append([x, y, width, height])

    return res


def insightface_to_haar(boxes: List[insightface.app.common.Face]) -> List[List[int]]:
    """

    """
    res = []
    for face in boxes:
        box = face.bbox.astype(int)
        x_l, y_t, x_r, y_b = box
        w = x_r - x_l
        h = y_b - y_t
        res.append([x_l, y_t, w, h])
    return res


def collect_yolo_ann(img_path: str) -> List[List[float]]:
    """

    :param img_path:
    :return:
    """
    lbl_path = f"{img_path[:-4].replace('images', 'labels')}.txt"
    with open(lbl_path, "r") as f:
        lines = f.readlines()
        expected_boxes = []
        for line in lines:
            box = line.strip()[2:].split()
            expected_boxes.append([float(i) for i in box])

    return expected_boxes


def test_haar_fr(test_data_path: str, model_name: str) -> List[str | floating]:
    """

    :param test_data_path:
    :return:
    """
    test_images = os.listdir(test_data_path)
    iou_scores = []
    accuracy_scores_05 = []
    accuracy_scores_07 = []
    for test_img in tqdm(test_images, desc="Testing haar cascade + face recognition model", ncols=100):
        path = f"{test_data_path}/{test_img}"
        img = cv2.imread(path)

        predictions = detect_faces_haar_cascade_recognition(img)
        expected_yolo = collect_yolo_ann(path)
        expected = yolo_to_haar(img, expected_yolo)

        # iou_score = iou(img, predictions, expected)
        # accuracy_score = calc_accuracy(img, predictions, expected)
        #
        # accuracy_scores.append(accuracy_score)
        # iou_scores.append(iou_score)

        iou_scores.append(iou(img, predictions, expected, f"data/{model_name}/{test_img}"))
        accuracy_scores_05.append(calc_accuracy(img, predictions.copy(), expected.copy(), 0.5))
        accuracy_scores_07.append(calc_accuracy(img, predictions.copy(), expected.copy(), 0.7))

    return [model_name, np.average(iou_scores).round(3), np.average(accuracy_scores_05).round(3), np.average(accuracy_scores_07).round(3)]


def test_yolo(test_data_path: str, model_path: str, model_name: str) -> list[str | floating]:
    """

    :param test_data_path:
    :return:
    """
    test_images = os.listdir(test_data_path)
    iou_scores = []
    accuracy_scores_05 = []
    accuracy_scores_07 = []
    for test_img in tqdm(test_images, desc=f"Testing YOLO model: ({model_name})", ncols=100):
        path = f"{test_data_path}/{test_img}"
        img = cv2.imread(path)
        model = YOLO(model_path)
        result = model(img, verbose=False)

        predictions = []
        for pred in result:
            # print(pred.boxes)
            boxes = pred.boxes
            for box in boxes:
                bb = box.xywhn
                predictions.append(bb[0])

        predictions = yolo_to_haar(img, predictions)
        expected_yolo = collect_yolo_ann(path)
        expected = yolo_to_haar(img, expected_yolo)

        # iou_score = iou(img, predictions, expected)
        # accuracy_score = calc_accuracy(img, predictions, expected)
        #
        # accuracy_scores.append(accuracy_score)
        # iou_scores.append(iou_score)

        iou_scores.append(iou(img, predictions, expected, f"data/{model_name}/{test_img}"))
        # print(expected)
        accuracy_scores_05.append(calc_accuracy(img, predictions.copy(), expected.copy(), 0.5))
        # print(expected)
        accuracy_scores_07.append(calc_accuracy(img, predictions.copy(), expected.copy(), 0.7))
    # return sum(iou_scores) / len(iou_scores), sum(accuracy_scores) / len(accuracy_scores)
    return [model_name, np.average(iou_scores).round(3), np.average(accuracy_scores_05).round(3), np.average(accuracy_scores_07).round(3)]


def test_insightface(test_data_path: str, model_name: str) -> list[str | floating]:
    test_images = os.listdir(test_data_path)
    iou_scores = []
    accuracy_scores_05 = []
    accuracy_scores_07 = []

    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    for test_img in tqdm(test_images, desc=f"Testing InsightFace", ncols=100):
        path = f"{test_data_path}/{test_img}"
        img = cv2.imread(path)

        faces = app.get(img)

        predictions = insightface_to_haar(faces)
        expected_yolo = collect_yolo_ann(path)
        expected = yolo_to_haar(img, expected_yolo)

        iou_scores.append(iou(img, predictions, expected, f"data/{model_name}/{test_img}"))
        accuracy_scores_05.append(calc_accuracy(img, predictions.copy(), expected.copy(), 0.5))
        accuracy_scores_07.append(calc_accuracy(img, predictions.copy(), expected.copy(), 0.7))

    return [model_name, np.average(iou_scores).round(3), np.average(accuracy_scores_05).round(3),
            np.average(accuracy_scores_07).round(3)]


if __name__ == '__main__':

    # path = "../data/Face detection.v1i.yolov8/test/images"
    # app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    # app.prepare(ctx_id=0, det_size=(640, 640))
    # for img_path in os.listdir(path):
    #     img = cv2.imread(f"{path}/{img_path}")
    #     faces = app.get(img)
    #     for face in faces:
    #         # print(bbox["bbox"])
    #         bbox = face.bbox.astype(int)
    #         print(bbox)
    #         cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    #     cv2.imshow("RetinaFace Detection", img)
    #     cv2.waitKey(0)







    results = []

    path = "../data/Face detection.v1i.yolov8/test/images"
    results.append(test_haar_fr(path, "haar_cascade + dlib (base model)"))
    # print(f"iou: {iou_score_haar_fr}, acc: {acc_score_haar_fr}")

    results.append(test_insightface(path, "insightface"))

    # model_path = "yolo_versions/v_yolov5s/weights/best.pt"
    model_path = "downloaded models/yolov8n-face.pt"
    results.append(test_yolo(path, model_path, "yolov8n-face"))
    # print(f"iou: {iou_score_yolo}, acc: {acc_score_yolo}")
    # model_path = "downloaded models/yolov8n-face.pt"

    model_path = "downloaded models/yolov11l-face.pt"
    results.append(test_yolo(path, model_path, "yolov11l-face"))

    model_path = "yolo_subset/v_yolov8s/weights/best.pt"
    results.append(test_yolo(path, model_path, "yolov8s"))
    # print(f"iou: {iou_score_yolo_t}, acc: {acc_score_yolo_t}")
    # acc = calc_accuracy(path, iou_score_yolo)

    df_res = pd.DataFrame(results, columns=["Model", "IoU (entire image)", "Accuracy (IoU ≥ 0.5)", "Accuracy (IoU ≥ 0.7)"])
    df_res.to_excel("results_face_detection.xlsx")




    # img_name = "0_Parade_marchingband_1_353"
    # img_path = "data/wider_face/val/images/0_Parade_marchingband_1_353.jpg"
    # lbl_path = "data/wider_face/val/labels/0_Parade_marchingband_1_353.txt"
    #
    # img = cv2.imread(img_path)
    #
    # with open(f"{lbl_path}", "r") as f:
    #     lines = f.readlines()
    #     expected_boxes = []
    #     for line in lines:
    #         box = line.strip()[2:].split()
    #         expected_boxes.append([float(i) for i in box])
    #
    # expected_boxes = yolo_to_haar(img, expected_boxes)
    #
    # detected_boxes = detect_faces_haar_cascade_recognition(img)
    # print(expected_boxes)
    # print(detected_boxes)
    #
    # iou = iou(img, detected_boxes, expected_boxes)
    # print(iou)



    # df = pd.read_parquet("../data/Embedding_On_The_Wall/full_df_clf_corrected.parquet")
    # # print(df["id"])
    # df = df[df["politician_in_img"].apply(lambda x: len(x) > 0)]
    # print(df)

    # os.makedirs("data/Embedding_On_The_Wall/", exist_ok=True)
    #
    # for img_id in tqdm(df["id"], desc="Running base model on test images"):
    #     img_name = f"{img_id}.jpg"
    #     # print(img_id in os.listdir("../data/Embedding_On_The_Wall/full_images"))
    #     if img_name in os.listdir("../data/Embedding_On_The_Wall/full_images"):
    #         # print(img_id)
    #         img_path = f"../data/Embedding_On_The_Wall/full_images/{img_name}"
    #         img_save_path = f"data/Embedding_On_The_Wall/{img_name}"
    #         img = cv2.imread(img_path)
    #
    #         annotations_on_img(img, img_save_path)




    # collect all images with politicians
    # os.makedirs("../data/Embedding_On_The_Wall/politicians/", exist_ok=True)
    # for img_id in tqdm(df["id"], desc="Running base model on test images"):
    #     img_name = f"{img_id}.jpg"
    #     img_path = f"../data/Embedding_On_The_Wall/full_images/{img_name}"
    #     img_dest = f"../data/Embedding_On_The_Wall/politicians/{img_name}"
    #     shutil.copy(img_path, img_dest)