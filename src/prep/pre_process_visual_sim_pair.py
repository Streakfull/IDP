import numpy as np
import pdb
from lutils.general import get_img_crop_from_frame, get_img_crop_from_frame_no_padding
from PIL import Image
import os


class PreProcessVisualSimPair():

    def __init__(self, f_a, f_b,
                 bench_mark_path="../logs/benchmarks/clip_1/corrected",
                 raw_frames_path="../logs/benchmarks/clip_1/raw_frames",
                 write_path="./raw_dataset/keypointpairs/debug"

                 ):
        self.f_a = f_a
        self.f_b = f_b
        self.bench_mark_path = bench_mark_path
        self.raw_frames_path = raw_frames_path
        self.write_path = write_path

    def get_pairs(self):
        dict_a = self.get_dict(self.f_a)
        dict_b = self.get_dict(self.f_b)
        if dict_a is None or dict_b is None:
            return
        positive_pairs = self.construct_positive_pairs(
            dict_a=dict_a, dict_b=dict_b)
        negative_pairs = self.construct_negative_pairs(
            dict_a, dict_b, len(positive_pairs)
        )

        if (len(positive_pairs) == 0 or len(negative_pairs) == 0):
            print("Returned")
            return
        self.write_pair(positive_pairs, dict_a, dict_b, True)
        self.write_pair(negative_pairs, dict_a, dict_b, False)

    def get_dict(self, f):
        labels_path = f"{self.bench_mark_path}/labels/frame_{f}.txt"
        features_path = f"{self.bench_mark_path}/featuresv2/frame_{f}.txt"
        image_path = f"{self.raw_frames_path}/frame_{f}.jpg"
        if not os.path.exists(labels_path) or not os.path.exists(features_path) or not os.path.exists(image_path):
            return None
        try:
            labels = np.loadtxt(f"{self.bench_mark_path}/labels/frame_{f}.txt")
            features = np.loadtxt(
                f"{self.bench_mark_path}/featuresv2/frame_{f}.txt")
            keypoints = np.loadtxt(
                f"{self.bench_mark_path}/keypoints/frame_{f}.txt")
            img = Image.open(f"{self.raw_frames_path}/frame_{f}.jpg")
            img = np.array((img))
            if labels.size == 0 or features.size == 0:
                return None
            inf_dict = {}
            for label in labels:
                x1, y1, x2, y2, conf, cls, id = label
                feat = np.zeros((384))
                kp = np.zeros(51)
                for f in features:
                    if (f[0] == id):
                        feat = f[1:]
                        break
                for keypoint in keypoints:
                    if (keypoint[0] == id):
                        kp = keypoint[1:]
                        break

                crop = get_img_crop_from_frame([x1, y1, x2, y2], img)
                inf_dict[f"{int(id)}"] = {"f": feat,
                                          "crop": crop,
                                          "kp": kp
                                          }

            return inf_dict
        except:
            return None

    def construct_positive_pairs(self, dict_a, dict_b):
        pairs = []
        for key in dict_a.keys():
            if (key in dict_b):
                pairs.append((key, key))
        return pairs

    def construct_negative_pairs(self, dict_a, dict_b, total_length):
        keys_a = np.array(list(dict_a.keys()))
        keys_b = np.array(list(dict_b.keys()))
        total_length = max(total_length, 2)
        negative_pairs = []
        loop_break = 0
        while len(negative_pairs) < total_length:
            # Randomly sample a key from each set
            key_a = np.random.choice(keys_a)
            key_b = np.random.choice(keys_b)

            # Ensure the pair is dissimilar
            if key_a != key_b:
                negative_pairs.append((key_a, key_b))
            if (loop_break >= 2*total_length):
                break
            loop_break += 1

        return negative_pairs

    def write_pair(self, pairs, dict_a, dict_b, is_similar):
        for pair in pairs:
            pa, pb = pair
            fa, fb = dict_a[pa]['f'], dict_b[pb]['f']
            imga, imgb = dict_a[pa]['crop'], dict_b[pb]['crop']
            kpa, kpb = dict_a[pa]['kp'], dict_b[pb]['kp']
            tgt = 1 if is_similar else 0
            folder_name = f"{self.write_path}/{self.f_a}-{self.f_b}-{pa}-{pb}-{tgt}"
            os.makedirs(folder_name, exist_ok=True)
            np.savetxt(os.path.join(folder_name, "x1.txt"), fa,
                       fmt="%.6f")  # Adjust precision if needed
            np.savetxt(os.path.join(folder_name, "x2.txt"), fb, fmt="%.6f")

            np.savetxt(os.path.join(folder_name, "x1k.txt"), kpa,
                       fmt="%.6f")  # Adjust precision if needed
            np.savetxt(os.path.join(folder_name, "x2k.txt"), kpb, fmt="%.6f")

            # Save PIL images as .jpg files
            imga.save(os.path.join(folder_name, "x1.jpg"))
            imgb.save(os.path.join(folder_name, "x2.jpg"))
