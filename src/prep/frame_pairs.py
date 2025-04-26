import random
import json
from tqdm import tqdm
import numpy as np
import os
from pytorchmodels.SiameseCompare import SiamaeseCompare


def create_number_pairs(max_distance,
                        num_pairs,
                        a_min=0,
                        a_max=999,
                        save_path=None

                        ):
    assert max_distance > 0, "Maximum distance must be greater than 0."
    assert num_pairs > 0, "Number of pairs must be greater than 0."

    numbers = np.arange(a_max+1)  # Numbers from 0 to 990
    half_pairs = num_pairs // 2
    pairs = []

    # Generate close pairs (inversely proportional to distance)
    for _ in range(half_pairs):
        num1 = np.random.choice(numbers)
        max_close_distance = max_distance // 2  # Close pairs have smaller range
        offset = np.random.randint(1, max_close_distance + 1)
        num2 = num1 + offset if num1 + offset <= a_max else num1 - offset
        pair = (min(num1, num2), max(num1, num2))
        pairs.append(pair)

    # Generate far pairs (proportional to distance)
    for _ in range(num_pairs - half_pairs):  # Remaining pairs
        num1 = np.random.choice(numbers)
        min_far_distance = max_distance // 2  # Far pairs have larger range
        offset = np.random.randint(min_far_distance, max_distance + 1)
        if num1 + offset <= a_max:
            num2 = num1 + offset
        elif num1 - offset >= 0:
            num2 = num1 - offset
        else:
            # Fallback if out of range
            num2 = np.random.choice(numbers)
        pair = (min(num1, num2), max(num1, num2))
        pairs.append(pair)

    if save_path:
      #  os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            for pair in pairs:
                f.write(f"{pair[0]} {pair[1]}\n")
        print(f"Pairs saved to {save_path}")

    return pairs


def read_pairs_from_file(file_path):
    pairs = []

    with open(file_path, 'r') as f:
        total_lines = sum(1 for _ in f)

    # Read the file with a tqdm progress bar
    with open(file_path, 'r') as f:
        for line in tqdm(f, total=total_lines, desc="Reading Pairs"):
            num1, num2 = map(int, line.strip().split(','))
            pairs.append((num1, num2))

    return pairs


def eval_pairs(pairs, save_path_logs=None, save_pth_json_matrix=None, cosine_threshold=0.2):
    sim_compare = SiamaeseCompare(eval_write_path=save_path_logs)
    tp, fp, fn, tn = 0, 0, 0, 0
    # pairs = [pairs[0]]
    for idx, pair in tqdm(enumerate(pairs), total=len(pairs), desc="Processing pairs"):
        a, b = pair
        frame_a = f"../logs/benchmarks/clip_1/raw_frames/frame_{a}.jpg"
        frame_b = f"../logs/benchmarks/clip_1/raw_frames/frame_{b}.jpg"
        c = sim_compare.process_pair_frames(
            frame_a, frame_b, cos_distance=cosine_threshold)
        matrix = c["confusion_matrix"]
        tp += matrix["tp"]
        fp += matrix["fp"]
        fn += matrix["fn"]
        tn += matrix["tn"]
        confusion_dict = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "idx": idx
        }
        if save_pth_json_matrix:
            os.makedirs(save_pth_json_matrix, exist_ok=True)
            file_path = os.path.join(
                save_pth_json_matrix, f"confusion_matrix.json")
            with open(file_path, 'w') as f:
                json.dump(confusion_dict, f, indent=4)
    return tp, fp, fn, tn


def select_far_and_close_pairs(pairs, num_pairs):
    # Calculate distances
    distances = [(pair, abs(pair[1] - pair[0])) for pair in pairs]

    # Sort by distance
    distances.sort(key=lambda x: x[1])

    # Categorize pairs into "close" and "far"
    num_total = len(distances)
    close_threshold = num_total // 3  # Bottom 30% are close
    far_threshold = num_total * 2 // 3  # Top 30% are far

    close_pairs = [dist[0] for dist in distances[:close_threshold]]
    far_pairs = [dist[0] for dist in distances[far_threshold:]]

    # Sample pairs
    num_far = int(num_pairs * 0.5)
    num_close = num_pairs - num_far

    selected_far = random.sample(far_pairs, min(num_far, len(far_pairs)))
    selected_close = random.sample(
        close_pairs, min(num_close, len(close_pairs)))

    # Combine and shuffle the selected pairs
    selected_pairs = selected_far + selected_close
    random.shuffle(selected_pairs)

    return selected_pairs
