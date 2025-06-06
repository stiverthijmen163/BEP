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
import zipfile
import gdown
from concurrent.futures import ThreadPoolExecutor, as_completed


def download_image(row: pd.Series, url_column: str, id_column: str, folder: str) -> bool:
    """
    Downloads an image from a URL provided a given location in a dataframe.

    :param row: row of a dataframe
    :param url_column: column name of the URL-column
    :param id_column: column name of the ID-column
    :param folder: folder to save the downloaded image to

    :return: whether the download was successful or not
    """
    # Collect the URL and save path
    url = row[url_column][2:-2]
    filename = f'{row[id_column]}.jpg'
    file_path = os.path.join(folder, filename)

    # Check if file already exists
    if os.path.isfile(file_path):
        return True  # Already exists

    # Try to download the image
    try:
        response = requests.get(url, timeout=10)  # Download image
        response.raise_for_status()  # Raise error if response is bad
        image = Image.open(io.BytesIO(response.content))  # Try to open image
        os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Create save path

        # Save the image to the desired location
        with open(file_path, "wb") as f:
            image.save(f, "JPEG")

        return True  # Success
    except Exception:
        return False  # Error


def download_img_from_url(dataframe: pd.DataFrame, url_column: str, id_column: str, folder: str, max_workers: int = 16) -> None:
    """
    Downloads images from all URL's in a dataframe.

    :param dataframe: dataframe containing all information regarding the images to download
    :param url_column: column name of the URL-column
    :param id_column: column name of the ID-column
    :param folder: folder to save the downloaded image to
    :param max_workers: maximum number of concurrent workers (used to speed up by dividing work over all cpu workers)
    """
    # Create save folder if needed
    os.makedirs(folder, exist_ok=True)

    # Initialize error count and list of function calls to execute
    error_count = 0
    tasks = []

    # Use multiple threads
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Collect all function calls to perform (all downloads)
        for _, row in dataframe.iterrows():
            task = executor.submit(download_image, row, url_column, id_column, folder)
            tasks.append(task)

        # Execute all function calls to download images
        for future in tqdm(as_completed(tasks), total=len(tasks), desc="Downloading images", ncols=100):
            # In case of an error, update error count
            if not future.result():
                error_count += 1

    print(f"Failed downloading {error_count} images")


def download_from_kaggle(kaggle_path: str, output_path: str, dataset_path: str) -> None:
    """
    Downloads a dataset from kaggle and moves it toe the desired output path.

    :param kaggle_path: name of kaggle dataset
    :param output_path: path to output folder
    :param dataset_path: path to the dataset where kagglehub downloads the data to
    """
    # Create output path if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Download latest version
    kagglehub.dataset_download(kaggle_path)

    # Set path to the cache where the data is stored
    cache_path = os.path.expanduser("~/.cache/kagglehub")

    # Move the data to the desired output folder
    for dataset in os.listdir(os.path.join(cache_path, "datasets", dataset_path)):
        shutil.move(os.path.join(cache_path, "datasets", dataset_path, dataset), os.path.join(output_path, dataset))

    # Remove used .cache
    shutil.rmtree(cache_path, ignore_errors=True)


def download_wider_face() -> None:
    """
    Download the WIDER-face dataset from huggingface and places it in the desired folders.
    """
    # Set the url's to download from
    urls = [
        "https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/ba0019b20b8c674d1f7bb8426f36ce69ea082648/data/WIDER_train.zip?download=true",
        "https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/ba0019b20b8c674d1f7bb8426f36ce69ea082648/data/WIDER_val.zip?download=true",
        "https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/ba0019b20b8c674d1f7bb8426f36ce69ea082648/data/WIDER_test.zip?download=true",
        "https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/ba0019b20b8c674d1f7bb8426f36ce69ea082648/data/wider_face_split.zip?download=true"
    ]

    # Set the temporary save folders
    save_folders_temp = [
        "data_temp/WIDER_train",
        "data_temp/WIDER_val",
        "data_temp/WIDER_test",
        "data_temp/wider_face_split"
    ]

    # Set the desired save folders
    save_folders = [
        "data/WIDER_face/train",
        "data/WIDER_face/val",
        "data/WIDER_face/test",
        "data/WIDER_face/wider_face_split"
    ]

    # Iterate over every dataset from WIDER face
    for url, temp_save_folder, save_folder in zip(urls, save_folders_temp, save_folders):
        # Collect the data
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("Content-Length", 0))
        chunk_size = 1024  # 1 KB

        # Load chunks into memory
        buffer = io.BytesIO()

        # Download dataset
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=f"Downloading {temp_save_folder.split('/')[-1]}", ncols=100) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:  # filter out keep-alive chunks
                    buffer.write(chunk)
                    pbar.update(len(chunk))

        # Go back to the start of the buffer
        buffer.seek(0)

        # Extract the downloaded zip file
        print(f"Extracting {temp_save_folder.split('/')[-1]}...")
        with zipfile.ZipFile(buffer) as z:
            z.extractall(temp_save_folder)

        # Set folder to move data from
        src_folder = f"{temp_save_folder}/{temp_save_folder.split('/')[-1]}"

        # Collect all file paths to move
        all_files = []
        for root, _, files in os.walk(src_folder):
            for file in files:
                file_path = f"{root}/{file}"
                all_files.append(file_path)

        # Move all files to the desired save folders
        for file in tqdm(all_files, ncols=100, desc=f"Moving data from {temp_save_folder.split('/')[-1]}"):
            # Create folders if needed
            os.makedirs(os.path.dirname(f"{save_folder}{file[len(src_folder):]}"), exist_ok=True)

            # Move the file
            shutil.move(file, f"{save_folder}{file[len(src_folder):]}")

        # Remove used .cache
        cache_path = os.path.expanduser("~/.cache/huggingface")
        shutil.rmtree(cache_path, ignore_errors=True)

    # Remove temporary folders
    shutil.rmtree("data_temp", ignore_errors=True)


def download_data_ruyters() -> None:
    """
    Downloads all images used by Ruyters.
    """
    # Download the parquet file containing all download links used by Ruyters
    url = "https://raw.githubusercontent.com/wieswies/embedding_on_the_wall/2cf675b321fab25fabd71f0ddc0f1079c7fe73b3/f__nos-nu-analysis/datasets/input/full_df_clf_corrected.parquet"
    response = requests.get(url)

    # Create save folder if needed
    if not os.path.exists("data/Embedding_On_The_Wall"):
        os.makedirs("data/Embedding_On_The_Wall")

    with open("data/Embedding_On_The_Wall/full_df_clf_corrected.parquet", "wb") as f:
        f.write(response.content)

    # Download the images used by Ruyters
    df = pd.read_parquet("data/Embedding_On_The_Wall/full_df_clf_corrected.parquet")
    # download_img_from_url("data/Embedding_On_The_Wall/full_images", df, "id", "img_link")
    download_img_from_url(df, "img_link", "id", "data/Embedding_On_The_Wall/full_images")


def main_download_data() -> None:
    """
    Main runner for downloading all data needed for this project.
    """
    # Remove all cache if there is any
    print("Removing .cache...")
    cache_path = os.path.expanduser("~/.cache/huggingface")
    shutil.rmtree(cache_path, ignore_errors=True)

    cache_path = os.path.expanduser("~/.cache/kagglehub")
    shutil.rmtree(cache_path, ignore_errors=True)

    # Remove all data
    print("Removing old data...")
    shutil.rmtree("data", ignore_errors=True)

    # Download the data used by Ruyters
    print("\nDownloading data used by Ruyters...")
    download_data_ruyters()

    # Download the manually annotated news images
    print("\nDownloading manually annotated news images...")
    gdown.download_folder(
        url="https://drive.google.com/drive/folders/1ErI645foO8fRMT2F46H0aDhFf6C14hRO?usp=drive_link",
        output="data")

    # Extract dat from zip file
    with zipfile.ZipFile("data/Face detection.v1i.yolov8.zip") as z:
        z.extractall("data/Face detection.v1i.yolov8")

    # Remove zip file
    os.remove("data/Face detection.v1i.yolov8.zip")

    # Download the WIDER-face dataset
    print("\nDownloading WIDER-face dataset...")
    download_wider_face()

    # Set kaggle datasets to download and their corresponding output paths
    kaggle_datasets = [
        "fareselmenshawii/face-detection-dataset",
        "vishesh1412/celebrity-face-image-dataset"
    ]

    output_paths = [
        "data/Face-Detection-Dataset (Kaggle)",
        "data/celebrity-face-image-dataset"
    ]

    dataset_paths = [
        "fareselmenshawii/face-detection-dataset/versions/3",
        "vishesh1412/celebrity-face-image-dataset/versions/1/Celebrity Faces Dataset"
    ]

    # Download every specified dataset from kaggle
    for k, o, d in zip(kaggle_datasets, output_paths, dataset_paths):
        print(f"\nDownloading {k.split('/')[-1]}...")
        download_from_kaggle(k, o, d)


if __name__ == "__main__":
    main_download_data()
