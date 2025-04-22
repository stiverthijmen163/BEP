import os
import gdown
import zipfile
import requests


def download_zip_from_drive(url: str, output: str) -> None:
    """
    Downloads zip-files from Google Drive.

    :param url: Google Drive URL to download zip file from
    :param output: Path to extract zip file to
    """
    # Download zip-file from Google Drive
    gdown.download(url, "temp.zip", fuzzy=True)

    # Create output direction if needed
    if not os.path.isdir(output):
        os.makedirs(output)

    # Extract zip-file to output folder
    print("Extracting zip file...\n")
    with zipfile.ZipFile("temp.zip") as z:
        z.extractall(output)

    # Remove temporary files
    os.remove("temp.zip")


def main_download_models() -> None:
    """
    Main runner for downloading models used and created with this code
    """
    # Set the Google Drive url's to download zip files from
    urls = [
        "https://drive.google.com/file/d/1wU1O_NYlB0GdWRp3zQ0b9s5Nc3B57lB3/view?usp=drive_link",  # YOLO small subset
        "https://drive.google.com/file/d/1O8aMdmPqoqdVpK3cgUpoXfui-TxIGZeg/view?usp=drive_link",  # YOLOv12-face
        "https://drive.google.com/file/d/1qXsQJ8ZT42_xSmWIYy85IcidpiZudOCB/view?usp=sharing"  # InsightFace
    ]

    # Set the output folder respectively
    output_folders = [
        "face_detection",
        "face_detection",
        os.path.expanduser("~/.insightface/models")
    ]

    # Download all data from GD
    for url, output_folder in zip(urls, output_folders):
        download_zip_from_drive(url, output_folder)


    # Set urls to download YOLO-face models from
    urls = [
        "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov11l-face.pt",
        "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov6n-face.pt"
    ]

    # Create output folder
    if not os.path.exists("face_detection/downloaded_models"):
        os.makedirs("face_detection/downloaded_models")

    # Download both models
    for url in urls:
        response = requests.get(url)

        # Write the data to the output folder
        with open(f"face_detection/downloaded_models/{url.split('/')[-1]}", "wb") as f:
            f.write(response.content)


if __name__ == "__main__":
    main_download_models()
