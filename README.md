# BEP
This GitHub is part of the Bachelor End Project regarding face detection and clustering models. In this code, all results can be checked and replicated (although that may not be recommended because of long run-times). Besides, the interactive visualization can be run using this code. For more information regarding the project, see the [thesis](https://github.com/stiverthijmen163/BEP/blob/main/Full_Thesis_BEP.pdf).

Below all steps are shown to run the full code and analysis. Besides, you will see whether the section is necessary to run to be able to use the visualization [Yes/No].

### Pre-requirements [Yes]
Download all necessary libraries from `requirements.txt`.
Note that you may need to download cmake from `cmake.org` if not installed on your device already, restart your device after installation.
In the case that some error appears while running the code, this can be because of the tensorflow version, try to install the `tensorflow==2.12.0` in that case.

### Downloading [Yes]
start by executing the following files: `download_data.py`, `download_models.py`.
The `download_data.py` file downloads all data necessary for this project.
The `download_models.py` file downloads all face detection used in this repository, this includes self-trained models like the YOLOv12s-face model.

## Face Detection
### Processing Data [No]
Execute `face_detection/process_data.py` to process all datasets needed for training, validating, and testing face detection models.

### Model Training [No]
Execute `face_detection/YOLO.py` to train some small models using all versions of the YOLOv12 architecture,
and train the final YOLOv12s-face model. Note: running this file takes up many hours and uses a lot of computing power, it is not advised to run this code especially without a GPU.

### Evaluation [No]
Execute the `face_detection/test.py` file to test all final models (the base model, InsightFace, YOLOv6n-face, YOLOv11l-face, YOLOv12s-face)
and save the resulting evaluation metrics. The results from the smaller models (validation metrics) can be found at [WandB](https://wandb.ai/t-m-a-broeren-eindhoven-university-of-technology/yolo_v12_small_subset?nw=nwusertmabroeren).

### Loss Comparison [No]
Execute the `face_detection/create_plot_loss_comparison.py` file to create the linechart containing the training and validation losses to check for overfitting in the YOLOv12s-face model.

## Face Clustering
### Pre-requirements [Yes]
To continue, make sure to update the numpy version to `numpy==1.23.5`.

### Process Data [No]
Execute the `face_clustering/process_data.py` file to process all data needed for testing face clustering algorithms.

### Face Embeddings [No]
Execute `face_clustering/calc_face_embeddings.py` to embed all faces using the 7 different models as explained in the thesis for both datasets (CFI & Embedding on the Wall).
Execute `face_clustering/explore_embeddings.py` to plot all actual clusters for all 7 different models for both datasets.

### Test Clustering Algorithms [No]
Execute `face_clustering/dbscan.py` to test the DBSCAN clustering algorithm with different parameters using all relevant embeddings models for both datasets.
Execute `face_clustering/chinese_whispers_model.py` to do the same using the Chinese Whispers algorithm. Moreover, both files print the running times for each data using the best embedding model (=face_recognition).

## Interactive Visualization [Yes]
Execute `visualization/index.py` to start the interactive visualization. It is advised to test all functionalities you want to use with a small set of images to ensure you have everything correctly setup.
For using the 'New Data' page, a csv-file has been created which is stored at `visualization/data/embedding_on_the_wall.csv`

```
Disclaimer:
This code is only tested on a Windows system and may not work on another system due to the code using the .cache locations from windows.
Moreover, this repository uses at least 30GB of disk space when every file has been executed.

Regarding the visualization: it has only been tested using a 1080p 19:6 screen, there may appear some glitches on other screens.
```