import cv2
from ultralytics import YOLO
from ultralytics.utils import LOGGER
import ultralytics
from roboflow import Roboflow
import os
from tqdm import tqdm
# import matplotlib.pyplot as plt
# import cv2
# import shutil
import wandb
import yaml
import pathlib


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


def create_yaml_file(yaml_path: str) -> None:
    """
    Creates a YAML file.
    :param yaml_path:
    :return:
    """
    doc = {
        "names": ["face"],
        "nc": 1,
        "train": "train/images",
        "val": "val/images",
        "path": f"{str(pathlib.Path().resolve())}/{'/'.join(yaml_path.split('/')[:-1])}",
    }

    with open(yaml_path, "w") as f:
        yaml.dump(doc, f)


def try_different_models():
    # models = ["yolov5s.pt", "yolov6s.yaml", "yolov8s.pt", "yolov9s.pt", "yolov10s.pt", "yolo11s.pt", "yolo12s.pt"]
    models = ["yolo12n.pt", "yolo12s.pt", "yolo12m.pt", "yolo12l.pt", "yolo12x.pt"]
    team_name = "t-m-a-broeren-eindhoven-university-of-technology"
    create_yaml_file("data/small_subset/data.yaml")
    project_name = "yolo_v12_small_subset"
    ultralytics.settings.update({"wandb": True})

    for m in models:
        model_name = m.split(".")[0]
        config = {
            "learning_rate": "auto",
            "architecture": model_name,
            "dataset": "small_subset",
            "epochs": 10,
            "imgsz": 640,
            "batch": 0.8
        }

        run_name = f"v_{model_name}"

        # Initialize wandb run
        wandb.init(
            entity=team_name,
            project=project_name,
            name=run_name,
            config=config
        )

        hyp = {
            "data": "data/small_subset/data.yaml",
            "epochs": 10,
            "imgsz": 640,
            "task": "detect",
            "device": 0,
            "plots": True,
            "batch": 0.8
            # "save_period": 1,
            # "patience": 10
        }

        # Log hyperparameters to wandb
        wandb.config.update(hyp)

        model = YOLO(m)
        results = model.train(
                **hyp,  # Hyperparameters
                project=wandb.run.project,
                name=wandb.run.name,
                exist_ok=True,
                tracker="wandb"
                # project="models",  # Name of the folder to store models in
                # name="YOLOv8s1280",  # Name of the run
                # data="data/small_subset/data.yaml"  # Path to the dataset
            )
        wandb.finish()


def train_final_yolo_model():
    team_name = "t-m-a-broeren-eindhoven-university-of-technology"
    create_yaml_file("data/subset/data.yaml")
    project_name = "yolo_v12s"
    ultralytics.settings.update({"wandb": True})

    model_name = "yolov12s-face"
    config = {
        "learning_rate": "auto",
        "architecture": model_name,
        "dataset": "subset",
        "epochs": 100,
        "imgsz": 640
    }

    # run_name = model_name

    # Initialize wandb run
    wandb.init(
        entity=team_name,
        project=project_name,
        name=model_name,
        config=config
    )

    hyp = {
        "data": "data/subset/data.yaml",
        "epochs": 100,
        "imgsz": 640,
        "task": "detect",
        "device": 0,
        "plots": True,
        "save_period": 1,
    }

    # Log hyperparameters to wandb
    wandb.config.update(hyp)

    model = YOLO("yolo12s.pt")
    results = model.train(
        **hyp,  # Hyperparameters
        project=wandb.run.project,
        name=wandb.run.name,
        exist_ok=True,
        tracker="wandb"
        # project="models",  # Name of the folder to store models in
        # name="YOLOv8s1280",  # Name of the run
        # data="data/small_subset/data.yaml"  # Path to the dataset
    )
    wandb.finish()


if __name__ == '__main__':
    # set_all_labels_to_face("data/kaggle-Face-Detection-Dataset/train/labels")
    # set_all_labels_to_face("data/kaggle-Face-Detection-Dataset/val/labels")

    # bounding_boxes_to_yolo("data/kaggle-Face-Detection-Dataset/train/images", "data/kaggle-Face-Detection-Dataset/train/labels")
    # bounding_boxes_to_yolo("data/kaggle-Face-Detection-Dataset/val/images", "data/kaggle-Face-Detection-Dataset/val/labels")

    # try_different_models()
    train_final_yolo_model()

    # LOGGER.info(f"WandB Enabled: {wandb.run is not None}")
    # ultralytics.settings.update({"wandb": True})
    # create_yaml_file("data/small_subset/data.yaml")
    #
    # config = {
    #     "learning_rate": "auto",
    #     "architecture": "YOLOv8s",
    #     "dataset": "small_subset",
    #     "epochs": 10,
    #     "imgsz": 640,
    # }
    #
    # team_name = "t-m-a-broeren-eindhoven-university-of-technology"
    # project_name = "yolo_versions"
    # run_name = "v_yolov8s"
    #
    # # Initialize wandb run
    # wandb.init(
    #     entity=team_name,
    #     project=project_name,
    #     name=run_name,
    #     config=config
    # )
    #
    # hyp = {
    #     "data": "data/small_subset/data.yaml",
    #     "epochs": 10,
    #     "imgsz": 640,
    #     "task": "detect",
    #     "device": 0,
    #     "plots": True,
    # }
    #
    # # Log hyperparameters to wandb
    # wandb.config.update(hyp)
    #
    # model = YOLO("yolov7.pt")
    # results = model.train(
    #         **hyp,  # Hyperparameters
    #         project=wandb.run.project,
    #         name=wandb.run.name,
    #         exist_ok=True,
    #         # tracker="wandb"
    #         # project="models",  # Name of the folder to store models in
    #         # name="YOLOv8s1280",  # Name of the run
    #         # data="data/small_subset/data.yaml"  # Path to the dataset
    #     )
    # wandb.finish()



    # Class
    # Images
    # Instances
    # Box(P
    # R
    # mAP50
    # mAP50 - 95): 100 % |██████████ | 101 / 101[00:25 < 00:00, 4.03
    # it / s]
    # all
    # 3225
    # 39671
    # 0.845
    # 0.573
    # 0.657
    # 0.361

    # for f in os.listdir("data/kaggle-Face-Detection-Dataset/val/images"):
    #     print(f)
    #     path = f"data/kaggle-Face-Detection-Dataset/labels2/{f.split('.jpg')[0]}.txt"
    #     print(path)
    #
    #     shutil.copy(path, f"data/kaggle-Face-Detection-Dataset/val/labels/{f.split('.jpg')[0]}.txt")


