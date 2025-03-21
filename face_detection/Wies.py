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


def detect_faces_haar_cascade_recognition():
    # HAAR cascade variables for face detection
    haar_cascade_path = 'haarcascade_frontalface_default.xml'
    haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + haar_cascade_path)

    img = cv2.imread("data/wider_face/val/images/0_Parade_Parade_0_913.jpg")

    # Convert to grayscale for Haar Cascade
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces using Haar Cascade
    haar_faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Detect faces using face_recognition
    face_locations = face_recognition.face_locations(img)
    fr_faces = [(left, top, right - left, bottom - top) for top, right, bottom, left in face_locations]

    # Combine and filter faces based on overlap
    all_faces = list(haar_faces) + fr_faces

    unique_faces = []
    for face in all_faces:
        if not any(overlap_ratio(face, uf) > 0.5 for uf in unique_faces):
            unique_faces.append(face)

    return unique_faces


if __name__ == '__main__':
    res = detect_faces_haar_cascade_recognition()
    print(res)
    print(len(res))
