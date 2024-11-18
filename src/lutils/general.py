from IPython.display import display
import os
import random
import numpy as np
import torch
from PIL import Image
import json
from cprint import *


def seed_all(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def write_tensor_text(tensor, path):
    with open(path, 'w') as f:
        for row in tensor:
            np.savetxt(f, row.unsqueeze(0).numpy(), fmt='%.4f')


def write_json(data, path):
    with open(path, 'w') as file:
        json.dump(data, file, indent=4, cls=NumpyArrayEncoder)


def create_directory(path):
    # Create the directory if it doesn't exist
    os.makedirs(path, exist_ok=True)
    # print(f"Directory '{path}' created or already exists.")


def write_tuple_file(path, data):
    with open(path, "w") as f:
        for integer, arr in data:
            # Convert array to space-separated string and write to file
            arr_str = " ".join(map(str, arr))
            f.write(f"{integer}, [{arr_str}]\n")


def show_image(image_path):
    """
    Display an image in a Jupyter notebook.

    Parameters:
    - image_path (str): The file path to the image you want to display.
    """
    # Load the image file
    image = Image.open(image_path)

    # Display the image in the notebook
    display(image)


def mkdir(path):
    if not os.path.exists(path):
        cprint.warn(f"- Creating new directory {path}")
        os.makedirs(path)
        return
    cprint.ok(f"- {path} directory found")


def get_img_crop_from_frame(box, frame, crop_size=(225, 225)):
    # Load the frame as a NumPy array
    img = frame
    img_height, img_width = img.shape[:2]

    # Calculate the center of the bounding box
    box_center_x = int((box[0] + box[2]) / 2)
    box_center_y = int((box[1] + box[3]) / 2)

    # Define the crop boundaries based on the center and crop size
    half_crop_width, half_crop_height = crop_size[0] // 2, crop_size[1] // 2
    left = max(0, box_center_x - half_crop_width)
    right = min(img_width, box_center_x + half_crop_width)
    top = max(0, box_center_y - half_crop_height)
    bottom = min(img_height, box_center_y + half_crop_height)

    # Crop the image
    crop = img[top:bottom, left:right]

    # Convert the cropped region back to a PIL Image and resize to ensure fixed size
    img = Image.fromarray(crop)
    img = img.resize(crop_size, Image.ANTIALIAS)

    return img
