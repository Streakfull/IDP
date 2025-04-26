from lutils.CreateStoredVideo import create_video_from_sequential_frames_6d
import os
from tqdm import tqdm


class CreateVideoFromFrames:
    def __init__(self, folder_list, fps=25):
        """
        Args:
            folder_list (list): List of folder paths, each containing a 'frames' directory.
            fps (int): Frames per second for the output video.
        """
        self.folder_list = folder_list
        self.fps = fps

    def process_folders(self, fps=25):
        """Processes each folder and creates a video from frames."""
        for folder in tqdm(self.folder_list, desc="Processing folders"):
            # frames_path = os.path.join(folder, "frames")
            frames_path = os.path.join(folder, "output_frames")
           # output_path = os.path.join(folder, "video.mp4")
            output_path = os.path.join(folder, "video.mp4")

            if os.path.exists(frames_path) and os.path.isdir(frames_path):
                create_video_from_sequential_frames_6d(
                    frames_path, output_path, fps=fps)
            else:
                print(f"Skipping {folder}: 'frames' directory not found.")


# folders_list = ["../logs/eval/custom-plz/try2-ball-filter"]

# frame_gen = CreateVideoFromFrames(folders_list)
# frame_gen.process_folders()
