import cv2
from ultralytics import YOLO
from roboflow import Roboflow
import os
from tqdm import tqdm
# import matplotlib.pyplot as plt
# import cv2
# import shutil


def load_yolo_annotations(annotations_path):
    """
    Load YOLO annotations from a file.
    Returns a list of (class_id, x_center, y_center, width, height).
    """
    boxes = []
    with open(annotations_path, "r") as file:
        for line in file:
            values = line.strip().split()
            # class_id = int(values[0])
            class_id = "_".join(values[:2])
            x_center, y_center, width, height = map(float, values[2:])
            boxes.append((class_id, x_center, y_center, width, height))
    return boxes


def set_all_labels_to_face(path_to_labels: str):
    """
    Sets all labels to 'fish' (0).
    :param path_to_labels: path to the labels to set to 0
    """
    # Iterate over all label files
    for f in tqdm(os.listdir(path_to_labels), desc="Setting labels to 0", ncols=100):
        # Collect all lines
        with open(f"{path_to_labels}/{f}", "r") as file:
            lines = file.readlines()

        # Modify the first element of each line
        modified_lines = []
        for line in lines:
            elements = line.split()[-4:]  # Split the line
            elements.insert(0, "0")  # set the first element to 0
            modified_line = ' '.join(elements)  # Join the line elements back
            modified_lines.append(modified_line)  # Append the result

        # Save the resulting labels
        with open(f"{path_to_labels}/{f}", 'w') as file:
            for line in modified_lines:
                file.write(line + '\n')
        file.close()


def bounding_boxes_to_yolo(path_to_images: str, path_to_labels: str):
    for f in tqdm(os.listdir(path_to_labels), desc="Converting to YOLO", ncols=100):
        # Load bounding boxes from the label file
        with open(f"{path_to_labels}/{f}", 'r') as file:
            bounding_boxes = [
                list(map(float, line.strip().split())) for line in file.readlines()
            ]

        img_path = f"{path_to_images}/{f[:-4]}.jpg"

        if os.path.isfile(img_path):
            img = cv2.imread(f"{path_to_images}/{f[:-4]}.jpg")
            frame_height, frame_width = img.shape[:2]
            bbs = []

            for box in bounding_boxes:
                class_id, x_center, y_center, width, height = box
                bb_x = x_center / frame_width
                bb_y = y_center / frame_height
                bb_w = width / frame_width
                bb_h = height / frame_height
                bbs.append(" ".join([str(int(class_id)), str(bb_x), str(bb_y), str(bb_w), str(bb_h)]))

            # Save the resulting labels
            with open(f"{path_to_labels}/{f}", 'w') as file:
                for line in bbs:
                    file.write(line + '\n')
            file.close()






if __name__ == '__main__':
    # set_all_labels_to_face("data/kaggle-Face-Detection-Dataset/train/labels")
    # set_all_labels_to_face("data/kaggle-Face-Detection-Dataset/val/labels")

    # bounding_boxes_to_yolo("data/kaggle-Face-Detection-Dataset/train/images", "data/kaggle-Face-Detection-Dataset/train/labels")
    # bounding_boxes_to_yolo("data/kaggle-Face-Detection-Dataset/val/images", "data/kaggle-Face-Detection-Dataset/val/labels")

    hyp = {
        "epochs": 10,
        "imgsz": 640,
        "task": "detect",
        "device": 0,
        "plots": True,
    }

    model = YOLO("yolov8s.pt")
    results = model.train(
            **hyp,  # Hyperparameters
            project="models",  # Name of the folder to store models in
            name="YOLOv8s",  # Name of the run
            data="data/kaggle-Face-Detection-Dataset//data.yaml"  # Path to the dataset
        )

    # for f in os.listdir("data/kaggle-Face-Detection-Dataset/val/images"):
    #     print(f)
    #     path = f"data/kaggle-Face-Detection-Dataset/labels2/{f.split('.jpg')[0]}.txt"
    #     print(path)
    #
    #     shutil.copy(path, f"data/kaggle-Face-Detection-Dataset/val/labels/{f.split('.jpg')[0]}.txt")


