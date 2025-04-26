import re
from tqdm import tqdm
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
    img = img.resize(crop_size, Image.Resampling.LANCZOS)

    return img


def get_img_crop_from_frame_no_padding(box, frame):
    """
    Extract the cropped image from the frame based on the bounding box.
    The crop will fit exactly to the bounding box dimensions without resizing.

    Args:
        box (tuple): The bounding box defined as (x_min, y_min, x_max, y_max).
        frame (numpy.ndarray): The image frame as a NumPy array.

    Returns:
        PIL.Image.Image: The cropped image as a PIL Image.
    """
    # Convert bounding box coordinates to integers
    left = int(max(0, box[0]))  # x_min
    top = int(max(0, box[1]))   # y_min
    right = int(min(frame.shape[1], box[2]))  # x_max
    bottom = int(min(frame.shape[0], box[3]))  # y_max

    # Crop the image
    crop = frame[top:bottom, left:right]

    # Convert the cropped region to a PIL Image
    img = Image.fromarray(crop)

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


def extract_frame_number(file_path):
    match = re.search(r'frame_(\d+)', file_path)
    return int(match.group(1)) if match else None


def find_object_by_id(objects, target_id):
    for obj in objects:
        if obj.id == target_id:
            return obj
    return None


def find_object_by_track_id(objects, target_id):
    for obj in objects:
        if obj.track_id == target_id:
            return obj
    return None


def create_image_grids(input_dir, output_dir, grid_size=32, pairs_per_row=4):
    """
    Creates grids of image pairs (bba_x and bbb_x) with a specified number of pairs per row and saves them.

    Args:
        input_dir (str): Directory containing the folders with bba and bbb images.
        output_dir (str): Directory to save the output grids.
        grid_size (int): Number of pairs per grid (default is 8).
        pairs_per_row (int): Number of pairs per row in the grid (default is 4).
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Collect and sort files
    bba_files = sorted([os.path.join(root, file) for root, _, files in os.walk(
        input_dir) for file in files if "bba_" in file])
    bbb_files = sorted([os.path.join(root, file) for root, _, files in os.walk(
        input_dir) for file in files if "bbb_" in file])

    # Pair files based on index
    pairs = list(zip(bba_files, bbb_files))

    # Process pairs into grids
    for grid_idx in range(0, len(pairs), grid_size):
        grid_pairs = pairs[grid_idx:grid_idx + grid_size]

        # Determine grid dimensions
        rows = (len(grid_pairs) + pairs_per_row - 1) // pairs_per_row
        pair_width, pair_height = Image.open(grid_pairs[0][0]).size
        grid_width = pair_width * 2 * pairs_per_row  # Each pair has 2 images
        grid_height = pair_height * rows

        # Create an empty canvas for the grid
        grid_image = Image.new("RGB", (grid_width, grid_height))

        # Add pairs to the grid
        for i, (bba_path, bbb_path) in enumerate(grid_pairs):
            row = i // pairs_per_row
            col = i % pairs_per_row

            # Get the images
            bba_img = Image.open(bba_path)
            bbb_img = Image.open(bbb_path)

            # Combine the pair side by side
            combined = Image.new("RGB", (pair_width * 2, pair_height))
            combined.paste(bba_img, (0, 0))
            combined.paste(bbb_img, (pair_width, 0))

            # Paste the combined image into the grid
            x_offset = col * pair_width * 2
            y_offset = row * pair_height
            grid_image.paste(combined, (x_offset, y_offset))

        # Save the grid
        grid_filename = os.path.join(
            output_dir, f"grid_{grid_idx // grid_size + 1}.png")
        grid_image.save(grid_filename)
        print(f"Saved grid: {grid_filename}")


def find_non_empty_logs(base_dir, fname="fn_log.txt"):
    """
    Searches through all directories within the given directory and finds folders
    that contain a non-empty 'fn_log.txt'.

    Args:
        base_dir (str): The path to the base directory to search.

    Returns:
        list: A list of parent folder names containing non-empty 'fn_log.txt'.
    """
    non_empty_folders = []
    # Walk through all directories and files in the base directory
    for root, dirs, files in os.walk(base_dir):
        # Check if 'fn_log.txt' exists in the current folder
        if fname in files:
            fn_log_path = os.path.join(root, fname)

            # Check if the file is non-empty
            if os.path.getsize(fn_log_path) > 0:
                # Add the parent folder name to the result
                parent_folder = os.path.basename(root)
                non_empty_folders.append(parent_folder)

    return non_empty_folders


def read_fn_logs_as_arrays(base_dir, folder_names):
    """
    Reads the 'fn_log.txt' files in the given folders and parses them as NumPy arrays.

    Args:
        base_dir (str): The base directory containing the folders.
        folder_names (list): A list of folder names to search for 'fn_log.txt'.

    Returns:
        dict: A dictionary where keys are folder names and values are NumPy arrays of tuples from 'fn_log.txt'.
    """
    fn_logs_data = {}

    for folder_name in folder_names:
        folder_path = os.path.join(base_dir, folder_name)
        fn_log_path = os.path.join(folder_path, 'fn_log.txt')

        # Check if the 'fn_log.txt' exists
        if os.path.exists(fn_log_path):
            try:
                # Read the file and parse as a NumPy array
                with open(fn_log_path, 'r') as f:
                    lines = f.readlines()
                    # Convert lines to tuples and store in NumPy array
                    data = np.array([eval(line.strip())
                                    for line in lines], dtype=np.int64)
                    fn_logs_data[folder_name] = data
            except Exception as e:
                print(f"Error reading file '{fn_log_path}': {e}")
                fn_logs_data[folder_name] = None
        else:
            print(f"Warning: 'fn_log.txt' not found in {folder_path}")
            fn_logs_data[folder_name] = None

    return fn_logs_data


def calculate_precision_recall(confusion_matrix_dir):
    methods_data = {"yoloVis": [], "vis": [], "yolo": []}

    for method in methods_data.keys():
        method_dir = os.path.join(confusion_matrix_dir, method)
        for threshold_folder in os.listdir(method_dir):
            threshold_dir = os.path.join(method_dir, threshold_folder)
            confusion_file = os.path.join(
                threshold_dir, "confusion_matrix", 'confusion_matrix.json')
            print(confusion_file, "FILE")
            # Check if the confusion_matrix.json exists
            if os.path.exists(confusion_file):
                with open(confusion_file, 'r') as f:
                    confusion_matrix = json.load(f)

                tp = confusion_matrix.get('tp', 0)
                fp = confusion_matrix.get('fp', 0)
                fn = confusion_matrix.get('fn', 0)
                tn = confusion_matrix.get('tn', 0)

                # Calculate Precision and Recall
                precision = tp / (tp + fp) if (tp + fp) != 0 else 1
                recall = tp / (tp + fn) if (tp + fn) != 0 else 0

                # Append data for each method at this threshold
                methods_data[method].append(
                    (float(threshold_folder), precision, recall))

    return methods_data
    import matplotlib.pyplot as plt


def plot_pr_curve(methods_data):
    plt.figure(figsize=(8, 6))

    # Plot for each method
    for method, data in methods_data.items():
        # Sort the data by recall (the second element in each tuple)
        sorted_data = sorted(data, key=lambda x: x[2])  # Sort by recall (x[2])

        # Unzip sorted data for plotting
        thresholds, precision, recall = zip(*sorted_data)

        # Plot the precision vs recall curve for this method
        plt.plot(recall, precision, label=method, marker='o')

    # Add labels and title
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Matching Pairwise Frames PR')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Show and save the plot
    plt.show()


def filter_unique_by_id(objects):
    """
    Filters a list of objects to keep only those with unique `id` values.

    Parameters:
        objects (list): List of dictionaries, each with an `id` key.

    Returns:
        list: A list of objects with unique `id` values.
    """
    unique_objects = []
    seen_ids = set()

    for obj in objects:
        if obj.id not in seen_ids:
            unique_objects.append(obj)
            seen_ids.add(obj.id)

    return unique_objects
