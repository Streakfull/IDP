import matplotlib.pyplot as plt
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


def map_to_conf_json(dir_files):
    return [f"conf-{x}.json" for x in dir_files]


def plot_pr_curves_from_directory(directory):
    # Lists to store precision and recall values for macro and micro methods
    macro_precisions = []
    macro_recalls = []
    micro_precisions = []
    micro_recalls = []

    # Iterate through all JSON files in the directory
    # import pdb
    # pdb.set_trace()
    dir_files = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    dir_files = map_to_conf_json(dir_files)
    # for filename in sorted(os.listdir(directory)):
    for filename in dir_files:
        if filename.endswith(".json"):  # Only process JSON files
            filepath = os.path.join(directory, filename)
            with open(filepath, "r") as f:
                data = json.load(f)
                # Append macro and micro precision and recall values
                macro_precisions.append(data["macro_prec"])
                macro_recalls.append(data["macro_rec"])
                micro_precisions.append(data["micro_prec"])
                micro_recalls.append(data["micro_recall"])

    # Plot Macro Precision-Recall Curve
    plt.figure(figsize=(10, 5))
    plt.plot(macro_recalls, macro_precisions,
             marker="o", label="Macro PR Curve")
    plt.title("Macro Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot Micro Precision-Recall Curve
    plt.figure(figsize=(10, 5))
    plt.plot(micro_recalls, micro_precisions,
             marker="o", label="Micro PR Curve")
    plt.title("Micro Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.legend()
    plt.show()


def extract_pr_values(conf_dir_map):
    """
    Extracts macro_prec and macro_rec values from .json files in the specified directories.

    Args:
        conf_dir_map (dict): A dictionary where keys are directory paths and values are labels.

    Returns:
        dict: A dictionary where keys are labels and values are lists of (macro_prec, macro_rec) tuples.
    """
    data = {}

    for directory, label in conf_dir_map.items():
        precision_recall = []

        dir_files = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        dir_files = map_to_conf_json(dir_files)
        dir_files.append("conf-f.json")

        # Sort files to maintain order (e.g., conf-0.1.json -> conf-1.json)
        for file in dir_files:
            if file.endswith('.json'):
                file_path = f"{directory}/{file}"

                # Read and parse JSON file
                try:
                    with open(file_path, 'r') as f:
                        content = json.load(f)
                        macro_prec = content.get('macro_prec')
                        macro_rec = content.get('macro_rec')

                        if macro_prec is not None and macro_rec is not None:
                            precision_recall.append((macro_prec, macro_rec))
                except:
                    print("Exception")

        # Store the results with the label
        if precision_recall:
            data[label] = precision_recall

    return data


def plot_pr_graph(data):
    """
    Plots the Precision-Recall (PR) graph for different methods.

    Args:
        data (dict): A dictionary where keys are labels and values are lists of (macro_prec, macro_rec) tuples.
    """
    plt.figure(figsize=(10, 6))

    for label, pr_values in data.items():
        precisions, recalls = zip(*pr_values)
        plt.plot(recalls, precisions, label=label, marker='o')

    plt.title('Precision-Recall (PR) Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)
    plt.show()


def f1_scores(precision_recall_data):
    """
    Calculate F1 scores for each method given precision-recall data.

    Args:
        precision_recall_data (dict): A dictionary where keys are labels and values are lists of (precision, recall) tuples.

    Returns:
        dict: A dictionary where keys are labels and values are lists of F1 scores.
    """
    f1_scores = {}
    for label, pr_values in precision_recall_data.items():
        scores = []
        for precision, recall in pr_values:
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0
            scores.append(f1)
        f1_scores[label] = scores
    return f1_scores


def plot_f1_scores(f1_scores):
    """
    Plot F1 scores for different methods.

    Args:
        f1_scores (dict): A dictionary where keys are labels and values are lists of F1 scores.
    """
    plt.figure(figsize=(10, 6))

    for label, scores in f1_scores.items():
        x_array = np.arange(len(scores))*10
        plt.plot(x_array, scores, marker='o', label=label)

    plt.title("F1 Scores for Different Methods")
    plt.xlabel("Confidence Threshold")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.grid()
    plt.show()
