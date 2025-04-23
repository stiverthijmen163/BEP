import os
import shutil

from ultralytics import YOLO
import cv2
from tqdm import tqdm
from typing import List


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
        face_img = img[int(y):int(y+h), int(x):int(x+w)]

        # Save the face
        cv2.imwrite(f"{save_path[:-4]}_{count}.jpg", face_img)

        count += 1


def process_images_celebrity(paths: List[str]) -> None:
    """
    Checks all images from the celebrity face dataset to cut out the correct face.

    :param paths: list of paths to images
    """
    # Load in the model to find all faces
    model = YOLO("../face_detection/yolo_v12s/yolov12s-face/weights/epoch60.pt")

    # Set the count of images which failed
    error_count = 0

    for img in tqdm(paths, ncols=100, desc="Collecting the correct faces"):
        # Read the image
        image = cv2.imread(img)

        # Collect the faces
        result = model(image, verbose=False)

        # Collect all bounding boxes
        predictions = []
        for pred in result:
            boxes = pred.boxes
            for box in boxes:
                bb = box.xywhn
                predictions.append(bb[0])

        # If there is more than one prediction get the one most centered
        if len(predictions) > 1:
            # Convert bounding boxes to desired format
            haar_pred = yolo_to_haar(image, predictions)

            # Calculate center of the image
            height, width = image.shape[:2]
            center_img = (width / 2, height / 2)

            dist = []

            # Find the distance from each face to the center
            for box in haar_pred:
                # Find the center of the face
                x, y, w, h = box
                center_box = (x + w / 2, y + h / 2)

                # Take minimum size into account
                if w * h > width * height * 0.02:
                    dist.append(((center_box[0] - center_img[0]) ** 2 + (center_box[1] - center_img[1]) ** 2) ** 0.5)

            # Save face if there exists a minimum distance to the center
            try:
                box = haar_pred[dist.index(min(dist))]
                save_cutout_faces(image, [box], "/".join(img.split("/")[1:]))
            except:  # Add to error counter otherwise
                error_count += 1
        # If there is exactly one face found, cut it out
        elif len(predictions) == 1:
            haar_pred = yolo_to_haar(image, predictions)
            save_cutout_faces(image, haar_pred, "/".join(img.split("/")[1:]))
        else:  # No faces found
            error_count += 1

    print(f"Failed for {error_count} images")


def main_process_data() -> None:
    """
    Main runner for pre-processing all data needed for testing different clustering method.
    """
    if os.path.exists("data"):
        shutil.rmtree("data")

    # Folder to find all images in
    p = "../data/celebrity-face-image-dataset"

    # Find all image paths in this folder
    image_paths = [
        os.path.join(root, file).replace("\\", "/")
        for root, _, files in os.walk(p)
        for file in files
        if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    # Process the data
    process_images_celebrity(image_paths)

    # Remove identified faces which are no faces
    ids = [
        "data/celebrity-face-image-dataset/Scarlett Johansson/077_776d5e0f_0.jpg",
        "data/celebrity-face-image-dataset/Scarlett Johansson/182_56820995_0.jpg",
        "data/celebrity-face-image-dataset/Tom Cruise/080_566ea9e9_0.jpg"
    ]

    for id in ids:
        if os.path.exists(id):
            os.remove(id)


if __name__ == "__main__":
    main_process_data()
