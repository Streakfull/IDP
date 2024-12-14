from tqdm import tqdm
from prep.construct_matches_dataset import ConstructMatchesDataset
from pytorchmodels.DeepSortObjectTrackingSiamese import DeepSortObjectTrackingSiamese
from prep.pre_process_visual_sim_pair import PreProcessVisualSimPair
import matplotlib.pyplot as plt
from pytorchmodels.SiameseCompare import SiamaeseCompare
from PIL import Image
from datasets.VisualSimMatches import VisualSimMatches
import cv2
import json
import os
import torch
from lutils.general import seed_all
import numpy as np
# %load_ext autoreload
# %autoreload 2

from pytorchmodels.DeepSortObjectTrackingSiamese import DeepSortObjectTrackingSiamese
# from pytorchmodels.SimpleObjectTracking import SimpleObjectTracking
from pytorchmodels.SiameseTracking import SiameseTracking
from lutils.general import seed_all, create_directory, write_tuple_file
from lutils.CreateStoredVideo import create_video_from_frames, create_video_from_sequential_frames
from metrics.CalcMetrics import CalcMetrics
from metrics.CalcMetrics import CalcMetrics
import torch
import numpy as np
import os
import json
import cv2
write_path = "../logs/tracking/manCityVsLiverpool/clip_1/week9/sort/frames"
labels_write_path = "../logs/tracking/manCityVsLiverpool/clip_1/week9/sort/labels"
write_path_deepsort = "../logs/tracking/manCityVsLiverpool/clip_1/week5/debug/frames"
video_path = "./raw_dataset/mancityVsLiverpool/clip_1.mp4"
write_path_crops = "../logs/tracking/manCityVsLiverpool/clip_1/week8/simple_tracking/crops"
bench_mark_path = "../logs/benchmarks/clip_1/corrected"
# %load_ext autoreload
# %autoreload 2
dataset_path = "./raw_dataset/footballPlayers/train/images"
bench_mark_path = "../logs/benchmarks/clip_1/corrected"
seed_all(111)


# def show_image(image_path):
#     """
#     Display an image in a Jupyter notebook.

#     Parameters:
#     - image_path (str): The file path to the image you want to display.
#     """
#     # Load the image file
#     image = Image.open(image_path)

#     # Display the image in the notebook
#     display(image)


sim_tracker = SiameseTracking(video_path, write_path)


def create():
    all_pairs = np.loadtxt("./raw_dataset/pairs_eval_clipped_20k.txt")
    all_pairs.shape
    i = 0
    for pair in tqdm(all_pairs, desc="preprocessing_pairs"):
        fa, fb = int(pair[0]), int(pair[1])

        prep = PreProcessVisualSimPair(
            fa, fb, write_path="./raw_dataset/frame_pairs_far2")
        prep.get_pairs()
        i += 1


def write_video():
    print("Writing video")
    sim_tracker = SiameseTracking(
        video_path, write_path, use_enhanced_tracking=False)
    deepsortTracker = DeepSortObjectTrackingSiamese(
        capture=video_path, write_path=write_path, use_kalman=True,
        use_visual_siamese=True, use_siamese=False

    )
    tracker = deepsortTracker
    y = tracker.process_video(
        video=video_path, write_path=write_path,
        start_frame=None,
        max_frames=None,
        labels_write_path=labels_write_path,
        print_cost_matrix=False
    )


if __name__ == "__main__":
    write_video()
