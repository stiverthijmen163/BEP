import re
from functions import *


# Create data folder if needed
if not os.path.exists("data"):
    os.mkdir("data")

# Load in the politics data
df = pd.read_parquet("../data/Embedding_On_The_Wall/full_df_clf_corrected.parquet")

# Use only those images which have a politician in them
data = df[df["politician_in_img"].apply(lambda x: len(list(x))) > 0].copy()
data = data[["id", "url", "category", "Issues_paragraphs"]].copy()

# Initialize list of images
images = []

# Collect all images
for _, row in data.iterrows():
    # Get the id of the image
    id = row["id"]

    # Read the image
    img = cv2.imread(f"../data/Embedding_On_The_Wall/full_images/{id}.jpg")

    # Transform image to base64 to save space in database
    img_b64 = img_to_base64(img)

    # Add to results
    images.append(img_b64)

# Add images to dataframe
data["img"] = images

# Find all items in the list in the remaining columns
data["category"] = data["category"].apply(lambda x: re.findall(r"'(.*?)'", x))
data["category"] = data["category"].apply(json.dumps)
data["Issues_paragraphs"] = data["Issues_paragraphs"].apply(lambda x: re.findall(r"'(.*?)'", x))
data["Issues_paragraphs"] = data["Issues_paragraphs"].apply(json.dumps)

# Save data to a csv file ready to use for the new data page
data.to_csv("data/embeddings_on_the_wall.csv", index=False)

