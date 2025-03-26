import numpy as np
import os
from io import BytesIO
from PIL import Image, UnidentifiedImageError
import face_recognition
import dlib
import cv2


def overlap_ratio(box1, box2):
    '''
    Function to calculate the overlap between two face-bounding-boxes detected in an image
    :param box1: First detected face
    :param box2: Second detected face
    '''
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2

    overlap = inter_area / float(box1_area + box2_area - inter_area)
    return overlap


def detect_faces_haar_cascade_recognition(img0: cv2.Mat):
    # HAAR cascade variables for face detection
    haar_cascade_path = 'haarcascade_frontalface_default.xml'
    haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + haar_cascade_path)

    # img = cv2.imread("data/wider_face/val/images/0_Parade_Parade_0_913.jpg")

    # Convert to grayscale for Haar Cascade
    gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

    # Detect faces using Haar Cascade
    haar_faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Detect faces using face_recognition
    face_locations = face_recognition.face_locations(img0)
    fr_faces = [(left, top, right - left, bottom - top) for top, right, bottom, left in face_locations]

    # Combine and filter faces based on overlap
    all_faces = list(haar_faces) + fr_faces

    print(f"Haar faces detected: {haar_faces}")
    print(f"face_recognition faces detected: {fr_faces}")

    unique_faces = []
    for face in all_faces:
        if not any(overlap_ratio(face, uf) > 0.5 for uf in unique_faces):
            unique_faces.append(face)

    return unique_faces


if __name__ == '__main__':
    img_name = "0_Parade_marchingband_1_353"
    img_path = "data/wider_face/val/images/0_Parade_marchingband_1_353.jpg"
    lbl_path = "data/wider_face/val/labels/0_Parade_marchingband_1_353.txt"
    img = cv2.imread(f"{img_path}")
    res = detect_faces_haar_cascade_recognition(img)
    print(res)
    print(len(res))

    with open(f"{lbl_path}", "r") as f:
        lines = f.readlines()
        expected_boxes = []
        for line in lines:
            box = line.strip()[2:].split()
            expected_boxes.append([float(i) for i in box])
    print(expected_boxes)

    # img = cv2.imread(f"{img_path}")
    frame_height, frame_width = img.shape[:2]
    print(frame_height, frame_width)

    boxes = []
    for box in expected_boxes:
        # img0 = cv2.imread("data/wider_face/val/images/0_Parade_Parade_0_913.jpg")
        x_c, y_x, w, h = box
        print(x_c, y_x, w, h)
        x = int((x_c - 0.5*w) * frame_width)
        y = int((y_x - 0.5*h) * frame_height)
        width = int(w * frame_height)
        height = int(h * frame_width)
        print(x, y, width, height)

        cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), thickness=2)
    # x, y, width, height = res[0]
    for box in res:
        x, y, width, height = box
        cv2.rectangle(img, (x, y), (x + width, y + height), (255, 0, 0), thickness=2)
    cv2.imshow('Detected faces', img)
    cv2.waitKey(0)

    # img_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)


