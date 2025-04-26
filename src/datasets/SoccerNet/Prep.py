import os
import cv2
import json
from tqdm import tqdm
from datasets.SoccerNet.constants import soccernet_dir, ball_action_dir


class Preprocess:
    def __init__(self, in_path=ball_action_dir, write_path="./raw_dataset/soccerNetV2/preprocessed"):
        self.in_path = in_path
        self.write_path = write_path

        # Create necessary directories
        self.create_directories()

    def create_directories(self):
        """Create the directory structure for train, valid, test."""
        for split in ['train', 'valid', 'test']:
            for folder in ['images', 'labels']:
                os.makedirs(os.path.join(self.write_path,
                            split, folder), exist_ok=True)

    def process_game(self, game_dir, split, max_frames=None, game_index=0):
        """Process each game and extract frames and annotations, with an optional frame limit."""
        video_file = '720p.mp4'  # Ensure the file name matches the dataset
        video_path = os.path.join(game_dir, video_file)

        if not os.path.exists(video_path):
            print(f"Warning: {video_file} not found in {game_dir}")
            return

        # Load video and get FPS
        video_data = cv2.VideoCapture(video_path)
        if not video_data.isOpened():
            print(f"Error: Unable to open video {video_path}")
            return

        fps = video_data.get(cv2.CAP_PROP_FPS)
        total_frames = int(video_data.get(cv2.CAP_PROP_FRAME_COUNT))

        try:
            with open(os.path.join(game_dir, 'Labels-ball.json'), 'r') as label_file:
                annotations = json.load(label_file)["annotations"]
        except FileNotFoundError:
            print(f"Error: Labels-ball.json not found in {game_dir}")
            return
        except json.JSONDecodeError:
            print(f"Error: Malformed Labels-ball.json in {game_dir}")
            return

        # Process annotations with tqdm for progress
        frame_count = 0
        for annotation in tqdm(annotations, desc=f"Processing {game_dir}", leave=False):
            if max_frames is not None and frame_count >= max_frames:
                break

            # Calculate frame index
            frame_index = round(float(annotation["position"]) * fps * 0.001)

            if frame_index >= total_frames:
                continue  # Skip invalid frame indices

            # Set video to the specific frame
            video_data.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = video_data.read()

            if not ret:
                print(
                    f"Warning: Failed to read frame {frame_index} in {video_path}")
                continue

            # Save the frame as an image
            image_filename = f"{game_index}_{frame_index}.png"
            sp = split.split("/")[0]
            image_path = os.path.join(
                self.write_path, sp, "images", image_filename)
            cv2.imwrite(image_path, frame)

            # Add game info to annotation and save it
            annotation['game'] = game_dir
            label_filename = f"{game_index}_{frame_index}.json"
            label_path = os.path.join(
                self.write_path, sp, "labels", label_filename)
            with open(label_path, 'w') as label_file:
                json.dump(annotation, label_file)

            frame_count += 1

        video_data.release()

    def preprocess(self, max_frames=None):
        """Main function to process all games and splits with tqdm for tracking."""
        for split in ['train/england_efl/2019-2020', 'valid/england_efl/2019-2020', 'test/england_efl/2019-2020']:
            split_dir = os.path.join(self.in_path, split)
            game_dirs = [d for d in os.listdir(
                split_dir) if os.path.isdir(os.path.join(split_dir, d))]
            game_index = 0

            # Progress bar for games
            for game_dir in tqdm(game_dirs, desc=f"Processing {split}", unit="game"):
                game_path = os.path.join(split_dir, game_dir)
                self.process_game(game_path, split, max_frames,
                                  game_index=game_index)
                game_index += 1


if __name__ == "__main__":
    # Example usage
    preprocessor = Preprocess()
    # Limit processing to 50 frames for debugging
    preprocessor.preprocess()
