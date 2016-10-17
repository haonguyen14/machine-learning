from PIL import Image
import numpy as np


def get_image(flat_array, w, h):

    array = np.reshape(flat_array, (3, w, h))
    array = np.transpose(array, (1, 2, 0))

    return Image.fromarray(array.astype(np.uint8), "RGB")


def normalize_image(image):

    max_val = np.max
    (
        np.abs(np.min(image)),
        np.abs(np.max(image))
    )

    scale = 0.0 if max_val < 1e-6 else (127.0 / max_val)
    offset = 128.0

    return image * scale + offset
