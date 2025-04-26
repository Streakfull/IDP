import torch.nn.functional as F
import torch
import os
import random
import configparser
from matplotlib.font_manager import weight_dict
import numpy as np
from tqdm import tqdm


class SoccerNetSamples():
    def __init__(self, samples_path, out_path, samples_per_game=15_000, max_number_of_games=None):
        self.samples_path = samples_path
        self.out_path = out_path
        self.samples_per_game = samples_per_game
        self.max_number_of_games = max_number_of_games
        self.games = self.set_number_of_games()

    def set_number_of_games(self):
        games = [os.path.join(self.samples_path, game) for game in os.listdir(self.samples_path)
                 if os.path.isdir(os.path.join(self.samples_path, game))]
        if self.max_number_of_games:
            games = games[:self.max_number_of_games]
        return games

    def total_samples(self):
        return len(self.games) * self.samples_per_game

    def get_all_samples(self):
        with open(self.out_path, 'w') as f:
            for game_path in tqdm(self.games, desc="Processing Games"):
                self.process_game(game_path)

    def process_game(self, game_path):
        seq_info_path = os.path.join(game_path, "seqinfo.ini")
        gt_path = os.path.join(game_path, "gt", "gt.txt")

        if not os.path.exists(seq_info_path) or not os.path.exists(gt_path):
            return

        config = configparser.ConfigParser()
        config.read(seq_info_path)
        max_length = int(config["Sequence"]["seqLength"])

        # Load only needed columns
        gt_data = np.loadtxt(gt_path, delimiter=",", usecols=range(6))

        self.process_positive_samples(gt_data, game_path)
        self.process_negative_samples(gt_data, game_path)

    def process_positive_samples(self, gt, game_path):
        num_positive = self.samples_per_game // 2
        close_samples = int(num_positive * 0.6)
        far_samples = num_positive - close_samples
        track_dict = {}

        for row in gt:
            frame, track_id = int(row[0]), int(row[1])
            if track_id not in track_dict:
                track_dict[track_id] = []
            track_dict[track_id].append(row)

        track_ids = list(track_dict.keys())
        random.shuffle(track_ids)

        for _ in range(close_samples):
            self.sample_matching_pair(track_dict, game_path, close=True)
        for _ in range(far_samples):
            self.sample_matching_pair(track_dict, game_path, close=False)

    def sample_matching_pair(self, track_dict, game_path, close=True):
        track_id = random.choice(list(track_dict.keys()))
        track_instances = track_dict[track_id]

        if len(track_instances) < 2:
            return

        track_instances.sort(key=lambda x: x[0])
        pivot = random.choice(track_instances)

        candidates = [x for x in track_instances if x[0] != pivot[0]]
        if not candidates:
            return

        distances = [abs(x[0] - pivot[0]) for x in candidates]
        if close:
            chosen = random.choices(candidates, weights=[
                                    1/d for d in distances], k=1)[0]
        else:
            chosen = random.choices(candidates, weights=distances, k=1)[0]

        self.add_pair(game_path, pivot, chosen, is_matching=1)

    def process_negative_samples(self, gt, game_path):
        num_negative = self.samples_per_game // 2
        close_samples = int(num_negative * 0.6)
        far_samples = num_negative - close_samples

        frame_dict = {}
        for row in gt:
            frame, track_id = int(row[0]), int(row[1])
            if frame not in frame_dict:
                frame_dict[frame] = []
            frame_dict[frame].append(row)

        frames = list(frame_dict.keys())
        random.shuffle(frames)

        for _ in range(close_samples):
            self.sample_non_matching_pair(frame_dict, game_path, close=True)
        for _ in range(far_samples):
            self.sample_non_matching_pair(frame_dict, game_path, close=False)

    def sample_non_matching_pair(self, frame_dict, game_path, close=True):
        frame = random.choice(list(frame_dict.keys()))
        frames_list_og = [f for f in frame_dict.keys() if f != frame]
        frames_list = np.array(frames_list_og, dtype=float)
        frame_distances = np.abs(frames_list-frame)

        if close:
            frame_distances = 1/frame_distances
        frame_distances /= frame_distances.sum()
        objects = frame_dict[frame]

        if len(objects) < 2:
            return

        pivot = random.choice(objects)

        target_frame = np.random.choice(frames_list_og, p=frame_distances)
        objects = frame_dict[target_frame]
        non_matching_candidates = [x for x in objects if x[1] != pivot[1]]

        if not non_matching_candidates:
            return

        def c(bb):
            x, y, w, h = bb[2], bb[3], bb[4], bb[5]
            cx = x + w / 2
            cy = y + h / 2
            return cx, cy

        def ed(point1, point2):
            return np.linalg.norm(np.array(point1) - np.array(point2))
        cp = c(pivot)

        distances = [ed(c(x), cp) for x in non_matching_candidates]
        if close:
            chosen = random.choices(
                non_matching_candidates, weights=distances, k=1)[0]
        else:
            distances = [1/(d+np.finfo(float).eps) for d in distances]
            chosen = random.choices(
                non_matching_candidates, weights=distances,  k=1)[0]

        self.add_pair(game_path, pivot, chosen, is_matching=0)

    def add_pair(self, game_path, x1, x2, is_matching):
        line = f"{game_path},{int(x1[0])},{int(x1[1])},{int(x1[2])},{int(x1[3])},{int(x1[4])},{int(x1[5])}," \
               f"{int(x2[0])},{int(x2[1])},{int(x2[2])},{int(x2[3])},{int(x2[4])},{int(x2[5])},{is_matching}\n"

        with open(self.out_path, 'a') as f:
            f.write(line)


class SoccerNetSamplesRandom():
    def __init__(self, samples_path, out_path, samples_per_game=15_000, max_number_of_games=None):
        self.samples_path = samples_path
        self.out_path = out_path
        self.samples_per_game = samples_per_game
        self.max_number_of_games = max_number_of_games
        self.games = self.set_number_of_games()

    def set_number_of_games(self):
        games = [os.path.join(self.samples_path, game) for game in os.listdir(self.samples_path)
                 if os.path.isdir(os.path.join(self.samples_path, game))]
        if self.max_number_of_games:
            games = games[:self.max_number_of_games]
        return games

    def total_samples(self):
        return len(self.games) * self.samples_per_game

    def get_all_samples(self):
        with open(self.out_path, 'w') as f:
            for game_path in tqdm(self.games, desc="Processing Games"):
                self.process_game(game_path)

    def process_game(self, game_path):
        seq_info_path = os.path.join(game_path, "seqinfo.ini")
        gt_path = os.path.join(game_path, "gt", "gt.txt")

        if not os.path.exists(seq_info_path) or not os.path.exists(gt_path):
            return

        config = configparser.ConfigParser()
        config.read(seq_info_path)
        max_length = int(config["Sequence"]["seqLength"])

        gt_data = np.loadtxt(gt_path, delimiter=",", usecols=range(6))

        self.process_random_samples(gt_data, game_path)

    def process_random_samples(self, gt, game_path):
        num_samples = self.samples_per_game
        frame_dict = {}
        track_dict = {}

        for row in gt:
            frame, track_id = int(row[0]), int(row[1])
            if frame not in frame_dict:
                frame_dict[frame] = []
            frame_dict[frame].append(row)
            if track_id not in track_dict:
                track_dict[track_id] = []
            track_dict[track_id].append(row)

        for _ in range(num_samples):
            if random.random() < 0.5:
                self.sample_random_pair(track_dict, game_path, is_matching=1)
            else:
                self.sample_random_pair(frame_dict, game_path, is_matching=0)

    def sample_random_pair(self, data_dict, game_path, is_matching):
        key = random.choice(list(data_dict.keys()))
        instances = data_dict[key]

        if len(instances) < 2:
            return

        x1, x2 = random.sample(instances, 2)
        if is_matching and x1[1] != x2[1]:
            return
        if not is_matching and x1[1] == x2[1]:
            return

        self.add_pair(game_path, x1, x2, is_matching)

    def add_pair(self, game_path, x1, x2, is_matching):
        line = f"{game_path},{int(x1[0])},{int(x1[1])},{int(x1[2])},{int(x1[3])},{int(x1[4])},{int(x1[5])}," \
               f"{int(x2[0])},{int(x2[1])},{int(x2[2])},{int(x2[3])},{int(x2[4])},{int(x2[5])},{is_matching}\n"
        # import pdb
        # pdb.set_trace()
        with open(self.out_path, 'a') as f:
            f.write(line)


class SoccerNetHardMiningTriplet():
    def __init__(self, samples_path, out_path, samples_per_game=15_000, max_number_of_games=None):
        self.samples_path = samples_path
        self.out_path = out_path
        self.samples_per_game = samples_per_game
        self.max_number_of_games = max_number_of_games
        self.games = self.set_number_of_games()

    def set_number_of_games(self):
        games = [os.path.join(self.samples_path, game) for game in os.listdir(self.samples_path)
                 if os.path.isdir(os.path.join(self.samples_path, game))]
        if self.max_number_of_games:
            games = games[:self.max_number_of_games]
        return games

    def total_samples(self):
        return len(self.games) * self.samples_per_game

    def get_all_samples(self):
        with open(self.out_path, 'w') as f:
            for game_path in tqdm(self.games, desc="Processing Games"):
                self.process_game(game_path)

    def process_game(self, game_path):
        seq_info_path = os.path.join(game_path, "seqinfo.ini")
        gt_path = os.path.join(game_path, "gt", "gt.txt")

        if not os.path.exists(seq_info_path) or not os.path.exists(gt_path):
            return

        config = configparser.ConfigParser()
        config.read(seq_info_path)
        max_length = int(config["Sequence"]["seqLength"])

        gt_data = np.loadtxt(gt_path, delimiter=",", usecols=range(6))

        self.process_triplets(gt_data, game_path)

    def process_triplets(self, gt, game_path):
        num_triplets = self.samples_per_game

        track_dict = {}
        for row in gt:
            frame, track_id = int(row[0]), int(row[1])
            if track_id not in track_dict:
                track_dict[track_id] = []
            track_dict[track_id].append(row)

        track_ids = list(track_dict.keys())
        random.shuffle(track_ids)

        for _ in tqdm(range(num_triplets), desc="Sampling Triplets", ncols=100):
            self.sample_triplet(track_dict, game_path)

    def sample_triplet(self, track_dict, game_path):
        track_id = random.choice(list(track_dict.keys()))
        track_instances = track_dict[track_id]

        if len(track_instances) < 2:
            return

        track_instances.sort(key=lambda x: x[0])
        anchor = random.choice(track_instances)

        # Select positive example (another instance of the same track_id but different frame)
        positive_candidates = [x for x in track_instances if x[0] != anchor[0]]
        if not positive_candidates:
            return

        positive = random.choice(positive_candidates)

        # Select negative example (different track_id, ideally a hard negative)
        negative = self.sample_hard_negative(track_dict, anchor, game_path)
        if negative is None:
            return

        self.add_triplet(game_path, anchor, positive, negative)

    def sample_hard_negative(self, track_dict, anchor, game_path):
        # Find a hard negative (track_id != anchor's track_id)
        other_track_ids = [
            track_id for track_id in track_dict if track_id != anchor[1]]
        random.shuffle(other_track_ids)

        for track_id in other_track_ids:
            negative_candidates = track_dict[track_id]
            random.shuffle(negative_candidates)

            # Select the most difficult negative (closest in feature space)
            hard_negative = self.find_hard_negative(
                anchor, negative_candidates)
            if hard_negative is not None:
                return hard_negative

        return None

    # def find_hard_negative(self, anchor, candidates, threshold=1.0):
    #     distances = [self.compute_distance(
    #         anchor, candidate) for candidate in candidates]
    #     min_distance = min(distances)
    #     hard_negative_idx = distances.index(min_distance)
    #     return candidates[hard_negative_idx] if min_distance < threshold else None
    def find_hard_negative(self, anchor, candidates, threshold=1.0):
        # Vectorized computation of distances
        anchor_center = np.array(anchor[2:6])  # x, y, w, h
        candidates_centers = np.array(
            [candidate[2:6] for candidate in candidates])  # x, y, w, h
        centers_diff = candidates_centers[:, :2] + candidates_centers[:,
                                                                      2:4] / 2 - anchor_center[:2] - anchor_center[2:4] / 2

        # Compute Euclidean distances
        distances = np.linalg.norm(centers_diff, axis=1)

        # Find the index of the closest candidate
        hard_negative_idx = np.argmin(distances)
        min_distance = distances[hard_negative_idx]

        if min_distance < threshold:
            return candidates[hard_negative_idx]
        else:
            # If no candidate satisfies the threshold, return the closest one
            return candidates[hard_negative_idx]

    def compute_distance(self, instance1, instance2):
        # Compute Euclidean distance between two bounding box centers or feature vectors
        x1, y1, w1, h1 = instance1[2:6]
        x2, y2, w2, h2 = instance2[2:6]
        center1 = (x1 + w1 / 2, y1 + h1 / 2)
        center2 = (x2 + w2 / 2, y2 + h2 / 2)
        return np.linalg.norm(np.array(center1) - np.array(center2))

    def add_triplet(self, game_path, anchor, positive, negative):
        # Save the triplet (anchor, positive, negative) to the output file
        line = f"{game_path},{int(anchor[0])},{int(anchor[1])},{int(anchor[2])},{int(anchor[3])},{int(anchor[4])},{int(anchor[5])}," \
               f"{int(positive[0])},{int(positive[1])},{int(positive[2])},{int(positive[3])},{int(positive[4])},{int(positive[5])}," \
               f"{int(negative[0])},{int(negative[1])},{int(negative[2])},{int(negative[3])},{int(negative[4])},{int(negative[5])}\n"

        with open(self.out_path, 'a') as f:
            f.write(line)


# # Root directory containing sequences
# samples_path = "./raw_dataset/soccernet-tracking/raw/tracking/train"
samples_path = "./raw_dataset/soccernet-tracking-test/raw/tracking/test"
# # Output file for storing positive & negative pairs
out_path = "./raw_dataset/soccernet-tracking/raw/tracking/test3-fixed.txt"
samples_per_game = 1_000  # Number of samples per game
max_number_of_games = None  # Set to limit the number of games processed

# # # Create an instance of the class
processor = SoccerNetSamples(
    samples_path, out_path, samples_per_game, max_number_of_games)

# # Generate the dataset
processor.get_all_samples()


def count_lines_ending_in_zero(file_path):
    with open(file_path, 'r') as f:
        count = sum(1 for line in f if line.strip().endswith('1'))
    return count


# Example usage
file_path = out_path
count = count_lines_ending_in_zero(file_path)
print(f"Number of lines ending with '0': {count}")
