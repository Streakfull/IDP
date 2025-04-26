import os
import shutil
import cv2
from tqdm import tqdm


def normalize_bbox(x, y, w, h, img_width, img_height):
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    w_norm = w / img_width
    h_norm = h / img_height
    return x_center, y_center, w_norm, h_norm


class SoccerNetPreprocessor:
    def __init__(self, train_path, test_path, output_path):
        self.train_path = train_path
        self.test_path = test_path
        self.output_path = output_path
        os.makedirs(os.path.join(
            output_path, "train", "images"), exist_ok=True)
        os.makedirs(os.path.join(
            output_path, "train", "labels"), exist_ok=True)
        os.makedirs(os.path.join(output_path, "test", "images"), exist_ok=True)
        os.makedirs(os.path.join(output_path, "test", "labels"), exist_ok=True)

    def process_split(self, split_path, split_type):
        clips = os.listdir(split_path)
        for clip in tqdm(clips, desc=f"Processing {split_type}"):
            clip_path = os.path.join(split_path, clip)
            gt_file = os.path.join(clip_path, "gt", "gt.txt")
            img_folder = os.path.join(clip_path, "img1")

            if not os.path.exists(gt_file) or not os.path.exists(img_folder):
                continue

            with open(gt_file, 'r') as f:
                annotations = [line.strip().split(',')
                               for line in f.readlines()]

            frame_dict = {}
            for ann in annotations:
                frame_idx, _, x, y, w, h, *_ = map(int, ann)
                frame_dict.setdefault(frame_idx, []).append((x, y, w, h))

            for frame_idx, bboxes in frame_dict.items():
                frame_name = f"{frame_idx:06d}.jpg"
                src_img_path = os.path.join(img_folder, frame_name)

                if not os.path.exists(src_img_path):
                    continue

                new_filename = f"{clip}_{frame_name}"
                dst_img_path = os.path.join(
                    self.output_path, split_type, "images", new_filename)
                dst_label_path = os.path.join(
                    self.output_path, split_type, "labels", new_filename.replace('.jpg', '.txt'))

                img = cv2.imread(src_img_path)
                img_height, img_width = img.shape[:2]

                with open(dst_label_path, 'w') as label_file:
                    for x, y, w, h in bboxes:
                        x_c, y_c, w_n, h_n = normalize_bbox(
                            x, y, w, h, img_width, img_height)
                        label_file.write(
                            f"0 {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")

                shutil.copy(src_img_path, dst_img_path)

    def process(self):
        self.process_split(self.train_path, "train")
        self.process_split(self.test_path, "test")
        print("Preprocessing complete.")


train_folder = "./raw_dataset/soccernet-tracking/raw/tracking/train"
test_folder = "./raw_dataset/soccernet-tracking-test/raw/tracking/test"
output_folder = "./raw_dataset/yolo_soccernet"

preprocessor = SoccerNetPreprocessor(train_folder, test_folder, output_folder)
preprocessor.process()
