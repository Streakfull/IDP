import numpy as np
from prep.pre_process_visual_sim_pair import PreProcessVisualSimPair
import os
from tqdm.notebook import tqdm


class ConstructMatchesDataset():

    def __init__(self, per_frame_count=40,
                 benchmark_path="../logs/benchmarks/clip_1/corrected",
                 max_frames=None,
                 write_path="./raw_dataset/frame_pairs_full_clipped_150_30_fixed",
                 clipping_window_frame=150
                 ):
        self.per_frame_count = per_frame_count
        self.benchmark_path = benchmark_path
        self.max_frames = max_frames
        self.write_path = write_path
        self.clipping_window_frame = clipping_window_frame

    def construct_dataset(self):
        filenames = os.listdir(f"{self.benchmark_path}/frames")
        numbers = [int(filename.split('_')[1].split('.')[0])
                   for filename in filenames]
        numbers = sorted(numbers)
        total = len(numbers) if self.max_frames is None else self.max_frames
        for i in tqdm(range(total), desc="Processing frames"):
            pivot = numbers[i]
            tgt = np.array(numbers[i+1:i+1+self.clipping_window_frame])
            if len(tgt) == 0:
                continue  # Skip if no target frames are available
            distance_arr = tgt-pivot

            f_match_count = int(self.per_frame_count/2)
            proportional_probs = distance_arr / np.sum(distance_arr)
            proportional_samples = np.random.choice(
                tgt, size=min(f_match_count, len(tgt)), replace=False, p=proportional_probs
            )
            inverse_probs = 1 / distance_arr
            inverse_probs /= np.sum(inverse_probs)
            inverse_samples = np.random.choice(
                tgt, size=min(f_match_count, len(tgt)), replace=False, p=inverse_probs
            )

            sampled_tgt = np.concatenate(
                (proportional_samples, inverse_samples))

            for tgt in sampled_tgt:
                # print(pivot, tgt, "TGT??")
                prep = PreProcessVisualSimPair(
                    (pivot), tgt, write_path=self.write_path)
                prep.get_pairs()
