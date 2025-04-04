import cv2
from tqdm import tqdm
import pandas as pd
from typing import List, Any
import os
import shutil
import random


def pixels_to_yolo(img: cv2.Mat, bb: List[Any]) -> List[str]:
    """
    Converts the following format to YOLO: x_center, y_center, width, height.

    :param img: the image to collect its size from
    :param bb: the bounding box coordinates to translate (assume float(<item from list>) is possible)

    :return: a list containing the class (set to 0 for face) and the annotation in YOLO format
    """
    # Check whether the list is of the expected format
    assert len(bb) == 4, f"length of bb_origin must be 4 but is {len(bb)}"

    # Collect width and height of the image
    frame_height, frame_width = img.shape[:2]

    # Collect items to translate to YOLO format
    x_center, y_center, width, height = bb

    # Calculate YOLO format annotations
    bb_x = float(x_center) / frame_width
    bb_y = float(y_center) / frame_height
    bb_w = float(width) / frame_width
    bb_h = float(height) / frame_height

    return ["0", str(bb_x), str(bb_y), str(bb_w), str(bb_h)] # label set to 0


def wider_face_collector(txt_loc: str) -> pd.DataFrame:
    """
    Collects all bounding boxes for the WIDER face dataset.

    :param txt_loc: path to txt file containing all bounding boxes and their corresponding image

    :return: a dataframe containing all images found in txt_loc and their corresponding bounding boxes
    """
    # Collect all lines from the txt file
    with open(txt_loc, "r") as f:
        lines = f.readlines()

    # Initialize variables
    count = 0
    bbs = []
    image_path = ""
    data = []

    # Iterate over all lines
    for line in lines:
        line = line.strip()
        if line.endswith(".jpg"):  # Line containing the image path
            if count != 0:  # Skip first iteration
                data.append([image_path, bbs])  # Append image and its corresponding bounding boxes

            # Start of new section in data
            image_path = line
            bbs = []
        elif len(line.split()) == 10 and not len([i for i in line.split() if int(i) == 0]) == 10:  # Lines containing a bounding box and ignore annotations of all zero's
            bbs.append(line)

        count += 1

    # Append last data
    data.append([image_path, bbs])

    # Create dataframe
    df = pd.DataFrame(data, columns=["image", "bbs"])
    return df


def wider_face_df_to_yolo(data: pd.DataFrame, train_val: str) -> None:
    """
    Convert all bounding boxes from the WIDER face dataset to YOLO format.

    :param data: the dataframe containing all bounding boxes and their corresponding image
    :param train_val: whether the dataset is train or validation data
    """
    # Iterate over all rows (images)
    for i, row in tqdm(data.iterrows(), total=len(data), desc="Transforming bounding boxes to YOLO format", ncols=100):
        # Open the image
        image_path = row["image"]
        image = cv2.imread(f"../data/WIDER_face/{train_val}/images/{image_path}")

        # Initialize bounding box variable
        bbs_yolo = []

        # Iterate over all annotations within the row
        for annotation in row["bbs"]:
            try:
                # Create list of all values
                bb_origin = [float(a) for a in annotation.split()]

                # Set x_min and y_min values to x_center and y_center
                bb_origin[0] += bb_origin[2] / 2
                bb_origin[1] += bb_origin[3] / 2

                # Convert to YOLO format
                bb_yolo = pixels_to_yolo(image, bb_origin[:4])
            except:  # In case of no annotation
                bb_yolo = []

            # Append YOLO format bounding box
            bbs_yolo.append(" ".join(bb_yolo))

        # Image file name
        new_path = image_path.split("/")[-1]

        # Write bounding boxes to the correct location
        with open(f"data/wider_face/{train_val}/labels/{new_path[:-4]}.txt", "w") as f:
            for line in bbs_yolo:
                f.write(line + '\n')
        f.close()

        # Copy the images to the correct locations
        shutil.copy(f"../data/WIDER_face/{train_val}/images/{image_path}", f"data/wider_face/{train_val}/images/{new_path}")


def wider_face() -> None:
    """
    Main runner for wider face dataset, accesses data and converts all annotations to YOLO format,
    also copies all data to the correct location.
    """
    # Set file locations
    wf_train_txt_loc = "../data/WIDER_face/wider_face_split/wider_face_train_bbx_gt.txt"
    wf_val_txt_loc = "../data/WIDER_face/wider_face_split/wider_face_val_bbx_gt.txt"

    # Get dataframes containing all images and their corresponding annotations
    df_train = wider_face_collector(wf_train_txt_loc)
    df_val = wider_face_collector(wf_val_txt_loc)

    # Convert all annotations to YOLO format and copy all data to the correct locations
    wider_face_df_to_yolo(df_train, "train")
    wider_face_df_to_yolo(df_val, "val")


def fdd_kaggle() -> None:
    """
    Main runner for wider face dataset, accesses data and copies all data to the correct location.
    """
    # Initializes path variables
    img_src_path = "../data/Face-Detection-Dataset (Kaggle)/images"
    lbl_src_path = "../data/Face-Detection-Dataset (Kaggle)/labels"
    dest_path = "data/Face-Detection-Dataset (Kaggle)"

    # Distinct in train and validation dataset
    for t in ["train", "val"]:
        # Initialize new path variables
        img_path = f"{img_src_path}/{t}"
        lbl_path = f"{lbl_src_path}/{t}"
        dest = f"{dest_path}/{t}"

        # Collect image paths
        images = os.listdir(img_path)

        # Copy images and labels to correct location
        copy_data(images, img_path, lbl_path, dest)


def exists_files(overwrite: bool) -> bool:
    """
    Checks whether all directories needed exist, assumes that all files exist if main folder exists.

    :param overwrite: whether to overwrite previous actions, thus removing all data and starting over

    :return: boolean whether all data processing should be run or not
    """
    # Check if all datasets are in the main data folder
    if not os.path.exists("../data/WIDER_face"):
        print("ASK TO DOWNLOAD DATA")

    # Initialize variable
    create = False

    # When 'overwrite', recreate file directories
    if overwrite:
        create = True
    else:  # Check if all destination folders exist
        if not os.path.exists("data/wider_face"):  # wider face dataset
            create = True
        elif not os.path.exists("data/Face-Detection-Dataset (Kaggle)"):  # Face-Detection-Dataset from Kaggle
            create = True

    # Create all directories needed for the face_detection section
    if create:
        make_dirs()

    return create


def make_dirs() -> None:
    """
    Creates all folders needed for processing the data.
    """
    # Remove all existing files
    print("Removing old files...")
    shutil.rmtree("data", ignore_errors=True)

    # Directories for wider face dataset
    print("Creating new directories...")
    os.makedirs("data/wider_face/train/images")
    os.makedirs("data/wider_face/train/labels")
    os.makedirs("data/wider_face/val/images")
    os.makedirs("data/wider_face/val/labels")

    # Directories for Face-Detection-Dataset from Kaggle
    os.makedirs("data/Face-Detection-Dataset (Kaggle)/train/images")
    os.makedirs("data/Face-Detection-Dataset (Kaggle)/train/labels")
    os.makedirs("data/Face-Detection-Dataset (Kaggle)/val/images")
    os.makedirs("data/Face-Detection-Dataset (Kaggle)/val/labels")


def copy_data(images: List[str], src_img: str, src_lbl: str, dest: str) -> None:
    """
    Copies images and their corresponding labels to new locations.

    :param images: list of image paths
    :param src_img: source image path
    :param src_lbl: source label path, labels are expected in txt files
    :param dest: destination path
    """
    for img in tqdm(images, desc="copying data", ncols=100):
        shutil.copy(f"{src_img}/{img}", f"{dest}/images/{img}")
        shutil.copy(f"{src_lbl}/{img[:-4]}.txt", f"{dest}/labels/{img[:-4]}.txt")


def main_process_data(overwrite: bool) -> None:
    """
    Main runner for processing all data needed for face detection.

    :param overwrite: whether to overwrite previous actions, thus removing all data and starting over
    """
    if exists_files(overwrite):
        print("Processing data from WIDER-face dataset...")
        wider_face()
        print("\nProcessing data from Face-Detection-Dataset from Kaggle...")
        fdd_kaggle()

    os.makedirs("data/small_subset/train/images")
    os.makedirs("data/small_subset/train/labels")
    os.makedirs("data/small_subset/val/images")
    os.makedirs("data/small_subset/val/labels")

    source = "data/Face-Detection-Dataset (Kaggle)/train/images"
    source_l = "data/Face-Detection-Dataset (Kaggle)/train/labels"
    dst = "data/small_subset/train"
    random.seed(123)
    img_train = random.sample(os.listdir(source), 500)
    copy_data(img_train, source, source_l, dst)

    source = "data/Face-Detection-Dataset (Kaggle)/val/images"
    source_l = "data/Face-Detection-Dataset (Kaggle)/val/labels"
    dst = "data/small_subset/val"
    random.seed(123)
    img_val = random.sample(os.listdir(source), 100)
    copy_data(img_val, source, source_l, dst)

    source = "data/wider_face/train/images"
    source_l = "data/wider_face/train/labels"
    dst = "data/small_subset/train"
    random.seed(123)
    img_train = random.sample(os.listdir(source), 500)
    copy_data(img_train, source, source_l, dst)

    source = "data/wider_face/val/images"
    source_l = "data/wider_face/val/labels"
    dst = "data/small_subset/val"
    random.seed(123)
    img_val = random.sample(os.listdir(source), 100)
    copy_data(img_val, source, source_l, dst)



    os.makedirs("data/subset/train/images")
    os.makedirs("data/subset/train/labels")
    os.makedirs("data/subset/val/images")
    os.makedirs("data/subset/val/labels")

    source = "data/Face-Detection-Dataset (Kaggle)/train/images"
    source_l = "data/Face-Detection-Dataset (Kaggle)/train/labels"
    dst = "data/subset/train"
    random.seed(123)
    img_train = random.sample(os.listdir(source), 2500)
    copy_data(img_train, source, source_l, dst)

    source = "data/Face-Detection-Dataset (Kaggle)/val/images"
    source_l = "data/Face-Detection-Dataset (Kaggle)/val/labels"
    dst = "data/subset/val"
    random.seed(123)
    img_val = random.sample(os.listdir(source), 500)
    copy_data(img_val, source, source_l, dst)

    source = "data/wider_face/train/images"
    source_l = "data/wider_face/train/labels"
    dst = "data/subset/train"
    random.seed(123)
    img_train = random.sample(os.listdir(source), 2500)
    copy_data(img_train, source, source_l, dst)

    source = "data/wider_face/val/images"
    source_l = "data/wider_face/val/labels"
    dst = "data/subset/val"
    random.seed(123)
    img_val = random.sample(os.listdir(source), 500)
    copy_data(img_val, source, source_l, dst)


if __name__ == '__main__':
    OVERWRITE = False
    main_process_data(OVERWRITE)