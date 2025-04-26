from datasets.SimMatches import SimMatches
from training.ModelTrainer_reid_latest import ModelTrainer
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
from pytorchmodels.DeepIOUTracker.DeepIOUTracker import DeepIOUTracker
from pytorchmodels.DeepIOUTracker.HMTracker import HMTracker
from ultralytics import YOLO
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
# write_path = "../logs/tracking/manCityVsLiverpool/clip_1/week10/kp-visual-bb/frames"
# labels_write_path = "../logs/tracking/manCityVsLiverpool/clip_1/week10/kp-visual-bb/labels"
write_path = "../logs/eval/finalpush/hmsort-kp-crops/frames"
labels_write_path = "../logs/eval/finalpush/hmsort-kp-crops/labels"
global_write_path = "../logs/eval/finalpush/hmsort-kp-crops/g"
write_path_deepsort = "../logs/tracking/manCityVsLiverpool/clip_1/week5/debug/frames"
video_path = "./raw_dataset/mancityVsLiverpool/clip_1.mp4"
write_path_crops = "../logs/tracking/manCityVsLiverpool/clip_1/week8/simple_tracking/crops"
bench_mark_path = "../logs/benchmarks/clip_1/corrected"
# frames_path = "./raw_dataset/soccernet-tracking/raw/tracking/train/SNMOT-060/img1"
# frames_path = "./raw_dataset/soccernet-tracking-test/raw/tracking/test/SNMOT-190/img1"
# frames_path = "./raw_dataset/soccernet-tracking-test/raw/tracking/test/SNMOT-190/img1"
# frames_path = "./raw_dataset/soccernet-tracking-test/raw/tracking/test/SNMOT-138/img1"
frames_path = "./raw_dataset/soccernet-tracking-test/raw/tracking/test/SNMOT-200/img1"
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


# sim_tracker = SiameseTracking(video_path, write_path)


def create():
    all_pairs = np.loadtxt("./raw_dataset/keypoint-paris-70k.txt")
    all_pairs.shape
    i = 0
    for pair in tqdm(all_pairs, desc="preprocessing_pairs"):
        fa, fb = int(pair[0]), int(pair[1])

        prep = PreProcessVisualSimPair(
            fa, fb, write_path="./raw_dataset/75k-close-far-norm")
        prep.get_pairs()
        i += 1


def write_video():
    torch.cuda.empty_cache()
    print("Writing video")
    sim_tracker = SiameseTracking(
        video_path, write_path, use_enhanced_tracking=True, track_queue_size=10)
    # deepsortTracker = DeepSortObjectTrackingSiamese(
    #     capture=video_path, write_path=write_path, use_kalman=True,
    #     use_visual_siamese=True, use_siamese=False

    # )
    tracker = sim_tracker
    y = tracker.process_video(
        video=video_path, write_path=write_path,
        start_frame=None,
        max_frames=None,
        labels_write_path=labels_write_path,

        # print_cost_matrix=False
    )


def write_video_frames_from_folder():
    torch.cuda.empty_cache()
    print("Writing video")
    sim_tracker = SiameseTracking(
        video_path, write_path, use_enhanced_tracking=False, track_queue_size=10)

    tracker = sim_tracker

    y = tracker.process_frames_from_folderg(
        frames_folder=frames_path,
        labels_write_path=labels_write_path,
        write_path=write_path,
        # max_frames=50
        # print_cost_matrix=False
    )


def write_video_frames_from_folder_deep_iou():
    torch.cuda.empty_cache()
    print("Writing video")
    # tracker = DeepIOUTracker(write_path)
    frames_path = "./raw_dataset/soccernet-tracking-test/raw/tracking/test/SNMOT-200/img1"

    folder_path = "../logs/eval/finalpush/debug"
    write_path = f"{folder_path}/frames"
    labels_write_path = f"{folder_path}/labels"
    global_text_file_path = folder_path
    tracker = HMTracker(write_path, use_high_res=True)

    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(write_path, exist_ok=True)
    os.makedirs(labels_write_path, exist_ok=True)

    y = tracker.process_frames_from_folderg(
        frames_folder=frames_path,
        labels_write_path=labels_write_path,
        write_path=write_path,
        global_text_file_path=global_text_file_path,
        max_frames=10
    )


def write_kp():
    seed_all(111)
    x = torch.cuda.mem_get_info()
    print(x)

    trainer = ModelTrainer(dataset_type=VisualSimMatches,
                           options={"tdm_notebook": True})
    dataset = trainer.data_loader_handler.dataset
    print("Dataset length: ", len(dataset))
    torch.cuda.empty_cache()
    print(torch.cuda.mem_get_info())
    model = trainer.model
    str(trainer.logger.experiment_dir)
    exp = f"{str(trainer.logger.experiment_dir)}/tb"
    exp
    #!tensorboard --logdir exp --|bind_all
    print(f"tensorboard --logdir {exp} --bind_all")
    model = YOLO("yolo11x-pose.pt")
    import pdb
    pdb.set_trace()
    dataloader = trainer.train_dataloader
    last_batch_index = len(dataloader) - 1
    last_batch = list(dataloader)[last_batch_index]
    with torch.no_grad():
        for batch in tqdm(trainer.train_dataloader):
            # import pdb;pdb.set_trace()
            x1, x2 = batch["x1k"], batch["x2k"]
            if any(el is None for el in x1) or any(el is None for el in x2):
                missing_x1 = [i for i, el in enumerate(x1) if el is None]

                missing_x2 = [i for i, el in enumerate(x2) if el is None]

                if missing_x1:
                    print(
                        f"Missing elements in x1 at indices: {batch['path'][missing_x1]}")

                if missing_x2:
                    print(
                        f"Missing elements in x2 at indices: {batch['path'][missing_x2]}")
                print("One or more elements in x1 or x2 are None.")

                x1_all_embed = model(batch["x1_path"], embed=[22])
                x2_all_embed = model(batch["x2_path"], embed=[22])
                for i in range(len(x1_all_embed)):
                    x1_write_path = f"{batch['path'][i]}/x1kv1.txt"
                    x2_write_path = f"{batch['path'][i]}/x1kv2.txt"
                    x1 = x1_all_embed[i]
                    x2 = x2_all_embed[i]
                    np.savetxt(x1_write_path, x1.cpu().numpy(),
                               fmt='%.6f')  # Save x1
                    np.savetxt(x2_write_path, x2.cpu().numpy(),
                               fmt='%.6f')  # Save x2
        print("CREATING X2")
        # for batch in tqdm(trainer.validation_dataloader):
        #     # import pdb;pdb.set_trace()
        #     x1_all_embed = model(batch["x1_path"], embed=[22])
        #     x2_all_embed = model(batch["x2_path"], embed=[22])
        #     for f in range(len(x1_all_embed)):
        #         x1_write_path = f"{batch['path'][f]}/x1kv1.txt"
        #         x2_write_path = f"{batch['path'][f]}/x1kv2.txt"
        #         x1 = x1_all_embed[f]
        #         x2 = x2_all_embed[f]
        #         np.savetxt(x1_write_path, x1.cpu().numpy(),
        #                    fmt='%.6f')  # Save x1
        #         np.savetxt(x2_write_path, x2.cpu().numpy(),
        #                    fmt='%.6f')  # Save x2


# Function to check directories for required files

def check_directories(paths, required_files):
    missing_count = 0
    missing_directories = []

    for path in paths:
        # Walk through each directory in the path
        for root, dirs, files in tqdm(os.walk(path), desc=f"Checking {path}"):
            # Check if all required files are present
            if not all(file in files for file in required_files):
                missing_count += 1
                missing_directories.append(root)

    # Print all directories with missing files
    print("\nDirectories with missing files:")
    for directory in missing_directories:
        print(directory)

    return missing_count


def check_dirs():
    # Define the dataset paths
    dataset_paths = [
        "./raw_dataset/latest/full/samples",
        "./raw_dataset/latest/full/clipped_text",
        "./raw_dataset/latest/full/clipped_manual_2",
    ]

    # Define the required files
    required_files = ["x1kv1.txt", "x1kv2.txt"]

    # Check directories and count missing ones
    missing_directories_count = check_directories(
        dataset_paths, required_files)

    # Output the result
    print(
        f"Total directories missing required files: {missing_directories_count}")


# def create_matching_ds():
#     k = ContstructMatchesDataset(
#         write_path="./raw_dataset/debug", max_frames=10)


if __name__ == "__main__":
    write_video_frames_from_folder_deep_iou()
    # write_kp()
    # create()
    # check_dirs()
