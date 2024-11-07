import os
import random
import numpy as np
import torch
from PIL import Image
import json


def seed_all(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def write_tensor_text(tensor, path):
    with open(path, 'w') as f:
        for row in tensor:
            np.savetxt(f, row.unsqueeze(0).numpy(), fmt='%.4f')


def write_json(data, path):
    with open(path, 'w') as file:
        json.dump(data, file, indent=4)


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
