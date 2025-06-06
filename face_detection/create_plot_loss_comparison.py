import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Open the data from the training of the YOLOv12s-face model
data = pd.read_csv("yolo_v12s/yolov12s-face/results.csv")

# Collect the information regarding losses
data_loss = data[["epoch", "train/dfl_loss", "val/dfl_loss"]].copy()
data_loss.columns = ["epoch", "training", "validation"]
data_loss = data_loss.set_index("epoch")

# Initialize the figure
plt.figure(figsize=(12, 6))

# Plot the training and validation loss
sns.set_style("darkgrid")
sns.lineplot(data_loss, dashes=False)

# Adjust the figure as preferred
plt.legend(loc="upper right", fontsize=16)
plt.title(f"Comparison of dfl-loss for training and validation set (YOLOv12s)", fontsize=20)
plt.xlabel("epoch", fontsize=16)
plt.ylabel("dfl-loss", fontsize=16)
plt.tick_params(axis="both", which="major", labelsize=14)
plt.xlim(0, 101)
plt.tight_layout()

# Save the figure
plt.savefig("yolo_v12s/yolov12s-face/W&B loss comparison.pdf")
