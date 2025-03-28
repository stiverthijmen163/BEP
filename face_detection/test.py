import pandas as pd
import pyarrow  # Needed for parquet files
import fastparquet  # Needed for parquet files
import os
import requests
from PIL import Image
import io
from tqdm import tqdm
from base_model import *
from typing import List
from ultralytics import YOLO
import matplotlib.pyplot as plt


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


def iou(img: cv2.Mat, boxes_det: List[List[int]], boxes_exp: List[List[int]]) -> float:
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

    tp = 0
    fp = 0
    fn = 0

    for i in range(frame_height):
        for j in range(frame_width):
            if mask1[i][j][0] == 1 and mask0[i][j][0] == 1:
                tp += 1
            elif mask1[i][j][0] == 0 and mask0[i][j][0] == 1:
                fp += 1
            elif mask1[i][j][0] == 1 and mask0[i][j][0] == 0:
                fn += 1

    # print(mask1[263][381])
    # print(frame_width, frame_height)
    # print(mask0)
    #
    # print(boxes_det)
    # print(boxes_exp)
    #
    # overlay_mask_cv2(img, mask0)
    # overlay_mask_cv2(img, mask1)

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
        width = int(w * frame_height)
        height = int(h * frame_width)

        res.append([x, y, width, height])

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


def test_haar_fr(test_data_path: str) -> float:
    """

    :param test_data_path:
    :return:
    """
    test_images = os.listdir(test_data_path)
    iou_scores = []
    for test_img in tqdm(test_images, desc="Testing haar cascade + face recognition model"):
        path = f"{test_data_path}/{test_img}"
        img = cv2.imread(path)

        predictions = detect_faces_haar_cascade_recognition(img)
        expected_yolo = collect_yolo_ann(path)
        expected = yolo_to_haar(img, expected_yolo)

        iou_score = iou(img, predictions, expected)

        iou_scores.append(iou_score)

    return sum(iou_scores) / len(iou_scores)


def test_yolo(test_data_path: str, model_path: str) -> float:
    """

    :param test_data_path:
    :return:
    """
    test_images = os.listdir(test_data_path)
    iou_scores = []
    for test_img in tqdm(test_images, desc="Testing YOLO model"):
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

        iou_score = iou(img, predictions, expected)

        iou_scores.append(iou_score)
    return sum(iou_scores) / len(iou_scores)


if __name__ == '__main__':
    path = "../data/Face detection.v1i.yolov8/test/images"
    # iou_score_haar_fr = test_haar_fr(path)
    # print(iou_score_haar_fr)

    # model_path = "yolo_versions/v_yolov5s/weights/best.pt"
    model_path = "yolo_face/yolov7-w6-face.pt"
    iou_score_yolo = test_yolo(path, model_path)
    print(iou_score_yolo)


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