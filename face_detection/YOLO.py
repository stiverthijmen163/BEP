from ultralytics import YOLO
import ultralytics
from roboflow import Roboflow
import wandb
import yaml
import pathlib


def create_yaml_file(yaml_path: str) -> None:
    """
    Creates a YAML file for training a YOLO model, assumes that the data is in the following folders:
    - train/images;
    - train/labels;
    - val/images;
    - val/labels.

    :param yaml_path: path of the yaml-file location
    """
    # Contents of the yaml-file
    doc = {
        "names": ["face"],  # Names of all classes
        "nc": 1,  # Number of classes
        "train": "train/images",  # Path to training images
        "val": "val/images",  # Path to validation images
        "path": f"{str(pathlib.Path().resolve())}/{'/'.join(yaml_path.split('/')[:-1])}",  # Path to dataset
    }

    # Write contents of the yaml-file
    with open(yaml_path, "w") as f:
        yaml.dump(doc, f)


def try_different_models() -> None:
    """
    Trains different YOLOv12 models on a small subset of the data and uploads to weights and biases (wandb).
    """
    models = ["yolo12n.pt", "yolo12s.pt", "yolo12m.pt", "yolo12l.pt", "yolo12x.pt"]  # Set models to train
    team_name = "t-m-a-broeren-eindhoven-university-of-technology"  # Set wandb to upload to correct location
    create_yaml_file("data/small_subset/data.yaml")  # Create yaml-file
    project_name = "yolo_v12_small_subset"  # Set folder-name to store the results in
    ultralytics.settings.update({"wandb": True})  # Make sure wandb logs all results

    # Iterate over all models
    for m in models:
        # Set name of the model
        model_name = m.split(".")[0]

        # Set model training configurations for wandb (has no effect on training)
        config = {
            "learning_rate": "auto",
            "architecture": model_name,
            "dataset": "small_subset",
            "epochs": 10,
            "imgsz": 640,
            "batch": 0.8
        }

        # Set name of the run
        run_name = f"v_{model_name}"

        # Initialize wandb run
        wandb.init(
            entity=team_name,
            project=project_name,
            name=run_name,
            config=config
        )

        # Set hyperparameters for YOLO training
        hyp = {
            "data": "data/small_subset/data.yaml",  # Path to dataset
            "epochs": 10,  # Number of training iterations
            "imgsz": 640,  # Resize images to 640 by 640
            "task": "detect",
            "device": 0,  # Use GPU for training
            "plots": True,  # Create plots after training
            "batch": 0.8  # Set batch size according to 80% GPU utilization
        }

        # Log hyperparameters to wandb
        wandb.config.update(hyp)

        # Load in the model
        model = YOLO(m)

        # Train the model
        results = model.train(
                **hyp,  # Hyperparameters
                project=wandb.run.project,
                name=wandb.run.name,
                exist_ok=True,
                tracker="wandb"
            )

        # Close wandb run
        wandb.finish()


def train_final_yolo_model() -> None:
    """
    Trains the final YOLO model used for face detection on a subset of all data.
    """
    team_name = "t-m-a-broeren-eindhoven-university-of-technology"  # Set wandb to upload to correct location
    create_yaml_file("data/subset/data.yaml")  # Create yaml-file
    project_name = "yolo_v12s"  # Set folder-name to store the results in
    ultralytics.settings.update({"wandb": True})  # Make sure wandb logs all results
    model_name = "yolov12s-face"  # Set name of the model

    # Set model training configurations for wandb (has no effect on training)
    config = {
        "learning_rate": "auto",
        "architecture": model_name,
        "dataset": "subset",
        "epochs": 100,
        "imgsz": 640
    }

    # Initialize wandb run
    wandb.init(
        entity=team_name,
        project=project_name,
        name=model_name,
        config=config
    )

    # Set hyperparameters for YOLO training
    hyp = {
        "data": "data/subset/data.yaml",  # Path to dataset
        "epochs": 100,  # Number of training iterations
        "imgsz": 640,  # Resize images to 640 by 640
        "task": "detect",
        "device": 0,  # Use GPU for training
        "plots": True,  # Create plots after training
        "save_period": 1,  # Save model weights every iteration
    }

    # Log hyperparameters to wandb
    wandb.config.update(hyp)

    # Load in the model
    model = YOLO("yolo12s.pt")

    # Train the model
    results = model.train(
        **hyp,  # Hyperparameters
        project=wandb.run.project,
        name=wandb.run.name,
        exist_ok=True,
        tracker="wandb"
    )

    # Close wandb run
    wandb.finish()


if __name__ == "__main__":
    try_different_models()
    train_final_yolo_model()
