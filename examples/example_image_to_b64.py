import numpy as np
import base64
import cv2

def img_to_base64(img: np.array) -> str:
    """
    Transforms an image to base64 string.

    :param img: image in np format to transform
    """
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")

# Read an image
img = cv2.imread("<PATH TO IMAGE>")

# Should look like:
# [[[ 39  36  28]
#   ...
#   [ 15  13  12]]
#
#   ...
#
#  [[12  29  48]
#   ...
#   [173  208  241]]]


# Encode image using base64
img_b64 = img_to_base64(img)

# Should look like:
# /9j/4AAQSkZJRgABAQAAAQABAAD/2w...