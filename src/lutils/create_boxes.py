import random
import shutil
import os
import cvzone
import cv2
import numpy as np
import pdb
from tqdm.notebook import tqdm


def plot_boxes(results, img, normalize=False):
    # Handle empty array
    if results.size == 0:
        print("No results to process.")
        return img
    # Ensure results is 2D
    if results.ndim == 1:
        results = np.expand_dims(results, axis=0)
    if not normalize:
        for box in results:
            cls, x1, y1, x2, y2, conf = box
            cls, x1, y1, x2, y2, conf = int(cls), int(x1), int(y1), int(
                x2), int(y2), round(float(conf), 2)
            w, h = x2-x1, y2-y1
            # pdb.set_trace()
            if (conf > 0):
                cvzone.cornerRect(img, (x1, y1, w, h), l=9,
                                  rt=1, colorR=(255, 0, 255))
                cvzone.putTextRect(
                    img, f'{conf}', (x1, y1-5), scale=1, thickness=1, colorR=(0, 0, 255))
        return img
    img_h, img_w, _ = img.shape

    for box in results:
        # Normalized coordinates
        cls, x1, y1, x2, y2 = box
        conf = 0.5

        # Denormalize coordinates
        x1, y1, x2, y2 = int(x1 * img_w), int(y1 *
                                              img_h), int(x2 * img_w), int(y2 * img_h)

        # Ensure other attributes are properly cast
        conf, cls = round(float(conf), 2), int(cls)
        w, h = x2 - x1, y2 - y1

        # Draw bounding boxes only if confidence > 0
        if conf > 0:
            cvzone.cornerRect(img, (x1, y1, w, h), l=3,
                              rt=1, colorR=(255, 0, 255))
            cvzone.putTextRect(
                img, f'{conf}', (x1, y1-5), scale=1, thickness=1, colorR=(0, 0, 255))
    return img


def plot_benchmark(video, write_path=None, labels_path=None, max_frames=None, start_frame=None, normalize=False):
    cap = cv2.VideoCapture(video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if (max_frames is not None):
        total_frames = max_frames
    frame = 0
    if (start_frame is not None):
        if (max_frames is not None):
            total_frames += start_frame

    with tqdm(total=total_frames-1, desc="Processing frames", unit="frame") as pbar:
        while True:
            _, img = cap.read()
            if (start_frame is not None and frame < start_frame):
                frame += 1
                pbar.update(1)
                continue

            labels = np.loadtxt(f"{labels_path}/frame_{frame}.txt")
            frames = plot_boxes(labels, img, normalize=normalize)
            cv2.imwrite(f"{write_path}/frame_{frame}.jpg", frames)
            frame += 1
            pbar.update(1)
            if frame >= total_frames:
                break


def shift_bounding_box(file_path, shift_x=0, shift_y=0, img_width=None, img_height=None):
    try:
        # Read the bounding box from the file
        with open(file_path, 'r') as file:
            lines = file.readlines()

        updated_lines = []
        for line in lines:
            if not line.strip():
                continue  # Skip empty lines

            # Parse the bounding box values
            cls, x_min, y_min, x_max, y_max, conf = map(
                float, line.strip().split())

            # Apply the shifts
            x_min += shift_x
            y_min += shift_y
            x_max += shift_x
            y_max += shift_y

            # Ensure coordinates remain within image bounds if image dimensions are provided
            if img_width is not None:
                x_min = max(0, min(x_min, img_width - 1))
                x_max = max(0, min(x_max, img_width - 1))
            if img_height is not None:
                y_min = max(0, min(y_min, img_height - 1))
                y_max = max(0, min(y_max, img_height - 1))

            # Append the updated bounding box
            updated_lines.append(
                f"{int(cls)} {int(x_min)} {int(y_min)} {int(x_max)} {int(y_max)} {round(conf,2)}\n")

        # Save the updated bounding boxes back to the same file
        with open(file_path, 'w') as file:
            file.writelines(updated_lines)

        print(f"Bounding box in {file_path} successfully shifted and updated.")
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")


def prop_bb(labels_dir, start_frame, end_frame, shift_x=0, shift_y=0):
    previous_label_path = None

    for frame in range(start_frame, end_frame + 1):
        current_label_path = os.path.join(labels_dir, f"frame_{frame}.txt")

        # First frame: Just ensure it exists, don't shift
        if frame == start_frame:
            if os.path.exists(current_label_path):
                previous_label_path = current_label_path
                print(
                    f"Frame {frame}: Using as base frame. No shifts applied.")
            else:
                print(
                    f"Frame {frame}: Label file not found! Cannot propagate bounding boxes.")
            continue

        # Propagate bounding boxes from the previous frame
        if not os.path.exists(current_label_path):
            print(
                f"Frame {frame}: Current label file not found. Using bounding box from previous frame.")
        os.system(f"cp {previous_label_path} {current_label_path}")

        # Shift the bounding box for the current frame
        shift_bounding_box(current_label_path, shift_x, shift_y)

        # Update the previous label path for the next iteration
        previous_label_path = current_label_path

    print("Processing complete.")


def filter_low_confidence(file_path, confidence_threshold, keep_below=False):
    """
    Processes bounding boxes in a label file based on a confidence threshold. 
    Either filters out bounding boxes below the threshold or keeps only those below it.

    Args:
        file_path (str): Path to the label file.
        confidence_threshold (float): Confidence threshold for processing bounding boxes.
        keep_below (bool): If True, keeps bounding boxes with confidence below the threshold.
                           If False, removes bounding boxes with confidence below the threshold.

    Returns:
        None
    """
    try:
        # Read the bounding box data from the file
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Process the bounding boxes based on the threshold and flag
        processed_lines = []
        for line in lines:
            if not line.strip():
                continue  # Skip empty lines

            # Parse the bounding box values
            values = list(map(float, line.strip().split()))
            if len(values) < 6:
                raise ValueError(
                    f"Invalid format in file: {file_path}. Expected format: cls x_min y_min x_max y_max conf")

            cls, x_min, y_min, x_max, y_max, conf = values

            # Determine whether to keep the bounding box
            if keep_below:
                # Keep only bounding boxes with confidence below the threshold
                if conf < confidence_threshold:
                    processed_lines.append(
                        f"{int(cls)} {x_min} {y_min} {x_max} {y_max} {conf:.2f}\n")
            else:
                # Keep only bounding boxes with confidence above the threshold
                if conf >= confidence_threshold:
                    processed_lines.append(
                        f"{int(cls)} {x_min} {y_min} {x_max} {y_max} {conf:.2f}\n")

        # Save the processed bounding boxes back to the same file
        with open(file_path, 'w') as file:
            file.writelines(processed_lines)

        operation = "kept below" if keep_below else "removed below"
        print(
            f"Updated {file_path}: Bounding boxes {operation} confidence {confidence_threshold}.")
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")


def normalize_labels(input_directory, output_directory, img_width, img_height):
    """
    Reads bounding box labels from a directory, normalizes the coordinates, removes confidence values,
    and writes the updated labels to a new directory.

    Args:
        input_directory (str): Path to the folder containing input label files.
        output_directory (str): Path to the folder to save normalized label files.
        img_width (int): Width of the images.
        img_height (int): Height of the images.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Get all .txt files in the input directory
    label_files = [f for f in os.listdir(
        input_directory) if f.endswith('.txt')]

    for label_file in label_files:
        input_file_path = os.path.join(input_directory, label_file)
        output_file_path = os.path.join(output_directory, label_file)

        try:
            # Load the file using numpy, skip empty files
            # Ensure data is at least 2D
            data = np.loadtxt(input_file_path, ndmin=2)
        except OSError:
            # If the file is empty or cannot be read, create an empty output file
            with open(output_file_path, 'w') as f:
                pass  # Create an empty file
            print(f"Created empty output file for: {label_file}")
            continue

        # Normalize the bounding box coordinates and remove confidence
        normalized_data = []
        for row in data:
            if len(row) != 6:
                print(f"Invalid line in {label_file}: {row}")
                continue
            cls, x1, y1, x2, y2, conf = row

            # Normalize coordinates
            x1_norm = x1 / img_width
            y1_norm = y1 / img_height
            x2_norm = x2 / img_width
            y2_norm = y2 / img_height

            # Append normalized data without confidence
            normalized_data.append([cls, x1_norm, y1_norm, x2_norm, y2_norm])

        # Save the normalized data to the output file
        np.savetxt(output_file_path, normalized_data,
                   fmt="%.6f", delimiter=" ")
        print(f"Processed and saved: {label_file}")


def create_dataset_with_splits(frames_dir, labels_dir, dataset_dir, train_ratio=0.9, valid_ratio=0.05, test_ratio=0.05):
    """
    Creates a dataset directory with train, valid, and test splits from raw frames and labels.

    Args:
        frames_dir (str): Directory containing raw image frames.
        labels_dir (str): Directory containing label files corresponding to the frames.
        dataset_dir (str): Directory to save the dataset splits.
        train_ratio (float): Proportion of data for the train split.
        valid_ratio (float): Proportion of data for the validation split.
        test_ratio (float): Proportion of data for the test split.
    """
    # Ensure the ratios sum up to 1
    assert abs(train_ratio + valid_ratio + test_ratio -
               1.0) < 1e-6, "Ratios must sum to 1."

    # Create output directories for train, valid, and test splits
    for split in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(dataset_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, split, 'labels'), exist_ok=True)

    # Get all frames and corresponding labels
    frames = sorted([f for f in os.listdir(frames_dir)
                    if f.endswith(('.jpg', '.png'))])
    labels = sorted([f for f in os.listdir(labels_dir) if f.endswith('.txt')])

    # Ensure the frames and labels are matched
    assert len(frames) == len(
        labels), "Number of frames and labels must match."
    for frame, label in zip(frames, labels):
        assert os.path.splitext(frame)[0] == os.path.splitext(
            label)[0], f"Mismatched frame and label: {frame} vs {label}"

    # Shuffle the data for randomness
    data = list(zip(frames, labels))
    random.shuffle(data)

    # Split the data into train, valid, and test sets
    total = len(data)
    train_split = int(total * train_ratio)
    valid_split = train_split + int(total * valid_ratio)

    train_data = data[:train_split]
    valid_data = data[train_split:valid_split]
    test_data = data[valid_split:]

    # Helper function to copy files
    def copy_files(data_split, split_name):
        for frame, label in data_split:
            shutil.copy(os.path.join(frames_dir, frame), os.path.join(
                dataset_dir, split_name, 'images', frame))
            shutil.copy(os.path.join(labels_dir, label), os.path.join(
                dataset_dir, split_name, 'labels', label))

    # Copy the files into their respective directories
    copy_files(train_data, 'train')
    copy_files(valid_data, 'valid')
    copy_files(test_data, 'test')

    print("Dataset splits created successfully:")
    print(f"Train: {len(train_data)} samples")
    print(f"Valid: {len(valid_data)} samples")
    print(f"Test: {len(test_data)} samples")


def normalize_labels_xyxy(input_directory, output_directory, img_width, img_height):
    """
    Converts bounding box labels from xyxy format to xywh, normalizes the coordinates,
    and writes the normalized labels to a new directory.

    Args:
        input_directory (str): Path to the folder containing input label files.
        output_directory (str): Path to the folder to save normalized label files.
        img_width (int): Width of the images.
        img_height (int): Height of the images.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Get all .txt files in the input directory
    label_files = [f for f in os.listdir(
        input_directory) if f.endswith('.txt')]

    for label_file in label_files:
        input_file_path = os.path.join(input_directory, label_file)
        output_file_path = os.path.join(output_directory, label_file)

        try:
            # Check if the file is empty before loading
            if os.path.getsize(input_file_path) == 0:
                # Create an empty output file if input is empty
                with open(output_file_path, 'w') as f:
                    pass
                print(f"Empty file processed: {label_file}")
                continue

            # Load the file with numpy
            data = np.loadtxt(input_file_path)
            # Ensure data is 2D (to handle cases with a single row of bounding boxes)
            data = np.atleast_2d(data)

        except Exception as e:
            print(f"Error reading file {label_file}: {e}")
            # Create an empty output file in case of error
            with open(output_file_path, 'w') as f:
                pass
            continue

        # Normalize the bounding box coordinates
        normalized_data = []
        for row in data:
            if len(row) != 6:
                print(f"Invalid row in {label_file}: {row}")
                continue
            cls, x_min, y_min, x_max, y_max, conf = row

            # Convert xyxy to xywh
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            box_width = x_max - x_min
            box_height = y_max - y_min

            # Normalize coordinates
            x_center_norm = x_center / img_width
            y_center_norm = y_center / img_height
            box_width_norm = box_width / img_width
            box_height_norm = box_height / img_height

            # Append normalized data
            normalized_data.append(
                [cls, x_center_norm, y_center_norm, box_width_norm, box_height_norm])

        # Save the normalized data to the output file
        if normalized_data:  # Only save if there's valid data
            np.savetxt(output_file_path, normalized_data,
                       fmt="%.6f", delimiter=" ")
        else:
            # Create an empty file if no valid data exists
            with open(output_file_path, 'w') as f:
                pass
        print(f"Processed: {label_file}")
