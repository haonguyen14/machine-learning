from PIL import Image
import numpy as np


def get_image(flat_array, w, h):

    array = np.reshape(flat_array, (3, w, h))
    array = np.transpose(array, (1, 2, 0))

    return Image.fromarray(array.astype(np.uint8), "RGB")
