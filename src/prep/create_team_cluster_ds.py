import numpy as np
from tqdm import tqdm
import os
from PIL import Image
import pdb
from lutils.general import get_img_crop_from_frame_no_padding


class CreateTeamClusterDs():

    def __init__(self,
                 benchmark_path="../logs/benchmarks/clip_1/corrected",
                 raw_frames_path="../logs/benchmarks/clip_1/raw_frames",
                 max_frames=None,
                 write_path="./raw_dataset/teams_clustering",
                 ):
        self.benchmark_path = benchmark_path
        self.max_frames = max_frames
        self.write_path = write_path
        self.bench_mark_path = benchmark_path
        self.raw_frames_path = raw_frames_path
        self.labels_path = f"{benchmark_path}/labels"

    def run(self):

        # os.makedirs(name=os.path.dirname(self.write_path), exist_ok=True)
        os.makedirs(self.write_path, exist_ok=True)
        raw_frames = os.listdir(self.raw_frames_path)
        labels = os.listdir(self.labels_path)
        total_crop_width = 0
        total_crop_height = 0
        crop_count = 0
        for label in tqdm(labels, desc="Processing frames"):
            frame_nr = label.split("_")[1].split(".txt")[0]
            labels = np.loadtxt(f"{self.labels_path}/{label}")
            crops, ids = self.get_crops_from_frame(labels, int(frame_nr))
            # Write crop here to self.write_path
            # The crop name should be based on the id corresponding to this index in the crops array
            # ALso gkeep track of running crops sizes and do avg

            for i, crop in enumerate(crops):
                crop_id = int(ids[i])
                crop_count += 1
                total_crop_width += crop.shape[1]
                total_crop_height += crop.shape[0]

                # Write the crop using the helper function
                self.write_crop(crop, crop_id, frame_nr)

    def get_crops_from_frame(self, labels, frame_nr):
        crops = []
        ids = []
        for label in labels:
            x1, y1, x2, y2, conf, clss, id = label
            box = [x1, y1, x2, y2]
            frame = Image.open(f"{self.raw_frames_path}/frame_{frame_nr}.jpg")
            frame = np.array(frame)
            crop = get_img_crop_from_frame_no_padding(box, frame)
            crop = np.array(crop)
            crops.append(crop)
            ids.append(id)
        return crops, ids

    def write_crop(self, crop, crop_id, frame_nr):
        """
        Saves the crop to disk with a file name based on crop ID and frame number.
        """
        crop_path = f"{self.write_path}/crop_{crop_id}_frame_{frame_nr}.jpg"
        Image.fromarray(crop).save(crop_path)
        # crop.save(crop_path)


if __name__ == "__main__":
    benchmark = CreateTeamClusterDs()
    benchmark.run()
