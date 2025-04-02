import pandas as pd
import pyarrow  # Needed for parquet files
import fastparquet  # Needed for parquet files
import os
import requests
from PIL import Image
import io
from tqdm import tqdm
import kagglehub
import shutil


def download_img_from_url(folder: str, dataframe: pd.DataFrame, id_column: str, url_column: str):
    """
    Downloads images from a list of URL's.

    :param folder: folder to which image must be written
    :param dataframe: dataframe in which the image urls are stored
    :param id_column: column that holds the unique identifier of the item, to be added to the filename
    :param url_column: column that holds the url information, which need to be opened and scraped
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    count = 0
    for i, row in tqdm(dataframe.iterrows(), total=dataframe.shape[0], desc="Downloading image"):
        url = row[url_column][2:-2]
        filename = f'{row[id_column]}.jpg'
        file_path = os.path.join(folder, filename)

        try:
            if not os.path.isfile(file_path):
                image_content = requests.get(url, timeout=10).content
                image_file = io.BytesIO(image_content)
                image = Image.open(image_file)

                with open(file_path, "wb") as f:
                    image.save(f, "JPEG")

        except Exception as e:
            count += 1

    print(f"Failed downloading {count} images")


def download_from_kaggle(kaggle_path: str, output_path: str) -> None:
    """

    """
    # Create output path if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Download latest version
    path = kagglehub.dataset_download(kaggle_path)
    shutil.move(path, output_path)


def main_download_data():
    """

    """
    # df = pd.read_parquet("data/Embedding_On_The_Wall/full_df_clf_corrected.parquet")
    # download_img_from_url("data/Embedding_On_The_Wall/full_images", df, "id", "img_link")

    kaggle_datasets = [
        "vishesh1412/celebrity-face-image-dataset"
    ]

    output_paths = [
        "data/celebrity-face-image-dataset"
    ]

    for k, o in zip(kaggle_datasets, output_paths):
        download_from_kaggle(k, o)


if __name__ == '__main__':
    main_download_data()
    # df = pd.read_parquet("data/Embedding_On_The_Wall/full_df_clf_corrected.parquet")
    # download_img_from_url("data/Embedding_On_The_Wall/full_images", df, "id", "img_link")

    # # Download latest version
    # path = kagglehub.dataset_download("vishesh1412/celebrity-face-image-dataset")
    # shutil.move(path, output_path)
    # print("Path to dataset files:", path)