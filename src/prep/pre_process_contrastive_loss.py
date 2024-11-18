import os
import numpy as np
import json
from lutils.general import write_json
from PIL import Image
from tqdm.notebook import tqdm


class PreProcessContrastiveLoss():

    def __init__(self, in_path,
                 write_path,
                 max_pairs,
                 write_path_features=None,
                 max_frames=None,
                 in_path_features=None,
                 frames_path=None
                 ) -> None:
        self.in_path = in_path
        self.write_path = write_path
        self.write_path_features = write_path_features
        self.frames = os.listdir(self.in_path)
        self.in_path_features = in_path_features
        self.max_frames = max_frames
        self.frames.sort()
        if (self.max_frames is not None):
            self.frames = self.frames[:max_frames]
        self.min_matching_dist = 20
        self.data_set_path = "./raw_dataset/match_pairs/eval"
        self.max_pairs = max_pairs
        self.frames_path = frames_path

    def add_features(self):
        for frame in self.frames:
            frame_id = self.get_frame_number(frame)
            bounding_boxes_gt = np.loadtxt(f"{self.in_path}/{frame}")
            folder = f"{frame_id}-{frame_id+1}"
            # import pdb
            # pdb.set_trace()
            if (frame_id >= 990):
                continue
            if (frame_id == 989):
                folder = f"{frame_id-1}-{frame_id}"
            modified_bounding_boxes = None
            all_features = self.load_vectors_from_file(
                f"{self.in_path_features}/{folder}/features_{frame_id}.txt")
            bounding_boxes_pred = np.loadtxt(
                f"{self.in_path_features}/{folder}/labels_{frame_id}.txt")

            for bounding_box_gt in bounding_boxes_gt:
                x1, y1, x2, y2, conf, cls, id = bounding_box_gt
                match_id, min_distance = self.find_min_distance(
                    bounding_box_gt, bounding_boxes_pred
                )

                if (min_distance > self.min_matching_dist):
                    continue
                feature_vector = next(
                    (feature[1] for feature in all_features if feature[0] == match_id), None)
                bounding_box_with_feature = np.hstack((
                    np.array([id]), feature_vector))[np.newaxis, :]
                try:
                    modified_bounding_boxes = np.vstack(
                        (modified_bounding_boxes, bounding_box_with_feature)) if modified_bounding_boxes is not None else bounding_box_with_feature
                except:
                    import pdb
                    pdb.set_trace()

            output_file = f"{self.write_path_features}/frame_{frame_id}.txt"
            if (modified_bounding_boxes is not None):
                np.savetxt(output_file, np.array(
                    modified_bounding_boxes), fmt="%f")

    def get_frame_number(self, frame):
        number = frame.split('_')[1].split('.')[0]
        number = int(number)
        return number

    def load_vectors_from_file(self, file_path):
        vectors = []
        with open(file_path, 'r') as file:
            for line in file:
                id_part, vector_part = line.split(',', 1)
                vector_id = int(id_part.strip())
                vector = np.fromstring(vector_part.strip()[1:-1], sep=' ')
                vectors.append((vector_id, vector))
        return vectors

    def get_bounding_box_center(self, x1, y1, x2, y2):
        w, h = x2-x1, y2-y1
        ret = np.asarray([x1, y1, w, h])
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def find_min_distance(self, gt_bb, all_pred):
        min_distance = np.Infinity
        match_id = None
        x1, y1, x2, y2, conf, cls, id = gt_bb
        center_gt = self.get_bounding_box_center(x1, y1, x2, y2)
        for pred_bb in all_pred:
            x1p, y1p, x2p, y2p, conf, cls, id_pred = pred_bb
            center_pred = self.get_bounding_box_center(x1p, y1p, x2p, y2p)
            distance = np.linalg.norm(center_pred-center_gt)
            if (distance < min_distance):
                min_distance = distance
                match_id = id_pred
        return match_id, min_distance

    def find_dist_arr(self, gt_bb, all_pred):
        x1, y1, x2, y2, = gt_bb
        center_gt = self.get_bounding_box_center(x1, y1, x2, y2)
        d = []
        for pred_bb in all_pred:
            x1p, y1p, x2p, y2p = pred_bb
            center_pred = self.get_bounding_box_center(x1p, y1p, x2p, y2p)
            distance = np.linalg.norm(center_pred-center_gt)
            d.append(distance)
        return np.array(d)

    def construct_pairs_dict(self):
        matches_dict = {}
        for frame in self.frames:
            bounding_boxes_gt = np.loadtxt(f"{self.in_path}/{frame}")
            frame_id = self.get_frame_number(frame)
            try:
                all_features = np.loadtxt(
                    f"{self.write_path_features}/{frame}")
            except:
                all_features = []
            if (len(all_features) == 0):
                continue
            for bounding_box_gt in bounding_boxes_gt:
                x1, y1, x2, y2, conf, cls, id = bounding_box_gt
                feature_vector = next(
                    (feature for feature in all_features if isinstance(feature, np.ndarray) and feature[0] == id), None)
                if feature_vector is None:
                    continue
                current_entry = matches_dict.get(id, None)
                if (current_entry is None):
                    matches_dict[id] = {}
                current_entry = matches_dict[id]

                matches_dict[id][frame_id] = {
                    "bb": np.array([x1, y1, x2, y2]),
                    "feat": feature_vector[1:],
                    "conf": conf
                }
        write_json(
            matches_dict, f"{self.data_set_path}/meta/matches_dict.json")
        return matches_dict

    def constuct_positive_pairs(self, input_dict, freq_dict, samples_path="samples", all_close=False, is_visual=False, all_random=False, all_far=False):
        object_ids = list(input_dict.keys())
        object_ids.sort()
        probs = [freq_dict[object_id]
                 for object_id in object_ids]
        if (all_random):
            probs = None
        chosen = {}
        i = 0
        with tqdm(total=self.max_pairs, desc="Processing Pairs", unit="pair") as pbar:
            while (i < self.max_pairs):
                while True:
                    object_id = np.random.choice(
                        object_ids, size=1, p=probs)[0]
                    if (object_id not in chosen):
                        chosen[object_id] = {}
                    occ = matches = input_dict[object_id]
                    occ = list(occ.keys())
                    occ.sort()
                    is_close = np.random.rand() <= 0.5 if all_close is False else True
                    if (all_far):
                        is_close = False
                    occ = np.array(occ)
                    x1 = pivot = np.random.choice(occ, size=1)[0]
                    occ = occ[occ != x1]
                    d_arr = np.abs(occ - pivot)
                    if (is_close):
                        d_arr = 1/d_arr

                    total = d_arr.sum()
                    p = d_arr/total
                    if (len(occ) <= 1):
                        break
                    if (all_random):
                        p = None
                    x2 = np.random.choice(occ, size=1, p=p)[0]
                    prev_obj_matches = chosen[object_id]
                    if (x1 not in prev_obj_matches):
                        prev_obj_matches[int(x1)] = []
                    if (x2 not in prev_obj_matches):
                        prev_obj_matches[int(x2)] = []
                    if (x1 in prev_obj_matches[int(x2)] or x2 in prev_obj_matches[int(x2)]):

                        # print("here??", x1, x2)
                        continue
                    else:
                        prev_obj_matches[int(x1)].append(x2)
                        prev_obj_matches[int(x2)].append(x1)
                        # break
                    if (is_visual):
                        self.write_visual_folder(
                            object_id, x1, x2, matches[x1], matches[x2], samples_path)
                    else:
                        x1f = self.construct_sample_from_dict(matches[x1])
                        x2f = self.construct_sample_from_dict(
                            matches[x2])
                        pair = np.vstack((x1f, x2f))
                        pair_name = f"{int(object_id)}-{x1}-{x2}-1"
                        np.savetxt(
                            f"{self.data_set_path}/{samples_path}/{pair_name}.txt", pair)
                        # print("HERE END?")
                    i += 1
                    pbar.update(1)
                    break

                # import pdb
                # pdb.set_trace()

    # def get_img_crop_from_frame(self, box, frame):
    #     img = np.array(Image.open(f"{self.frames_path}/frame_{frame}.jpg"))
    #     crop = img[int(box[1]): int(
    #         box[3]), int(box[0]): int(box[2])]
    #     img = Image.fromarray(crop)
    #     img = img.resize((225, 225), Image.ANTIALIAS)
    #     return img

    def get_img_crop_from_frame(self, box, frame, crop_size=(225, 225)):
        # Load the frame as a NumPy array
        img = np.array(Image.open(f"{self.frames_path}/frame_{frame}.jpg"))
        img_height, img_width = img.shape[:2]

        # Calculate the center of the bounding box
        box_center_x = int((box[0] + box[2]) / 2)
        box_center_y = int((box[1] + box[3]) / 2)

        # Define the crop boundaries based on the center and crop size
        half_crop_width, half_crop_height = crop_size[0] // 2, crop_size[1] // 2
        left = max(0, box_center_x - half_crop_width)
        right = min(img_width, box_center_x + half_crop_width)
        top = max(0, box_center_y - half_crop_height)
        bottom = min(img_height, box_center_y + half_crop_height)

        # Crop the image
        crop = img[top:bottom, left:right]

        # Convert the cropped region back to a PIL Image and resize to ensure fixed size
        img = Image.fromarray(crop)
        img = img.resize(crop_size, Image.ANTIALIAS)

        return img

    def construct_sample_from_dict(self, sample):
        bb = sample["bb"]
        ft = sample["feat"]
        conf = np.array(sample["conf"])
        return np.hstack((bb, conf, ft))

    def construct_appearance_frequency_dict(self, input_dict):
        object_ids = input_dict.keys()
        freq_dict = {}
        for object_id in object_ids:
            freq_dict[object_id] = len(input_dict[object_id].keys())
        values_total = np.array(list(freq_dict.values())).sum()

        for id in freq_dict.keys():
            freq_dict[id] = freq_dict[id]/values_total
        write_json(
            freq_dict, f"{self.data_set_path}/meta/freq_dict.json")
        values_total = np.array(list(freq_dict.values())).sum()
        return freq_dict

    def construct_negative_pairs(self, input_dict, freq_dict, samples_path="samples", all_close=False, is_visual=False):
        object_ids = list(input_dict.keys())
        object_ids.sort()
        probs = [freq_dict[object_id] for object_id in object_ids]
        i = 0
        with tqdm(total=self.max_pairs, desc="Processing Pairs", unit="pair") as pbar:
            while (i < self.max_pairs):
                x1_mid, x2_mid = np.random.choice(
                    object_ids, size=2, p=probs, replace=False)
                assert x1_mid != x2_mid
                is_close = np.random.rand() <= 0.5
                euclidean = np.random.rand() <= 0.5
                occx1 = matchesx1 = input_dict[x1_mid]
                occx2 = matchesx2 = input_dict[x2_mid]
                occx1 = list(occx1.keys())
                occx2 = list(occx2.keys())
                occx1.sort()
                occx2.sort()
                occx1 = np.array(occx1)
                occx2 = np.array(occx2)
                x1 = pivot = np.random.choice(occx1, size=1)[0]
                if (euclidean == False):
                    occx2 = occx2[occx2 != pivot]
                    d_occx2 = np.abs(occx2-pivot)
                    if (is_close):
                        d_occx2 = 1/d_occx2
                    total = d_occx2.sum()
                    p = d_occx2/total
                    if (len(occx2) <= 1):
                        continue
                    x2 = np.random.choice(occx2, size=1, p=p)[0]
                    x2_dict = matchesx2[x2]
                    x2f = self.construct_sample_from_dict(matchesx2[x2])
                    pair_name = f"{int(x1_mid)}d{int(x2_mid)}-{x1}-{x2}-0"
                else:
                    frame_ids = []
                    bb_arr = []
                    for object_id in object_ids:
                        frames = input_dict[object_id]
                        if (x1 in frames):
                            frame_ids.append(object_id)
                    for k in range(len(frame_ids)):
                        oid = frame_ids[k]
                        bb = input_dict[oid][x1]["bb"]
                        bb_arr.append(bb)

                    bb_arr = np.array(bb_arr)
                    frame_ids = np.array(frame_ids)
                    pivot_index = np.where(frame_ids == x1_mid)[0][0]
                    pivot_bb = bb_arr[pivot_index]
                    bb_arr = np.delete(bb_arr, pivot_index, axis=0)
                    d_arr = self.find_dist_arr(pivot_bb, bb_arr)
                    frame_ids = np.delete(frame_ids, pivot_index)
                    is_close_frames = is_close if all_close is False else True
                    if (is_close_frames):
                        d_arr = 1/d_arr
                    total = d_arr.sum()
                    p = d_arr/total
                    if (len(frame_ids) <= 1):
                        continue
                    x2_preid = np.random.choice(frame_ids, size=1, p=p)[0]
                    # import pdb
                    # pdb.set_trace()
                    x2_dict = input_dict[x2_preid][x1]
                    x2 = x1
                    x2f = self.construct_sample_from_dict(
                        x2_dict)
                    pair_name = f"{int(x1_mid)}s{x2_preid}-{x1}-{x1}-0"

                if (is_visual):
                    x1_dict = matchesx1[x1]
                    self.write_visual_folder(
                        x1_mid, x1, x2,
                        x1_dict=x1_dict,
                        x2_dict=x2_dict,
                        is_positive=False, x2_id=x2_mid,
                        samples_path=samples_path
                    )
                else:
                    x1f = self.construct_sample_from_dict(matchesx1[x1])
                    pair = np.vstack((x1f, x2f))
                    np.savetxt(
                        f"{self.data_set_path}/{samples_path}/{pair_name}.txt", pair)
                i += 1
                pbar.update(1)

    def construct_negative_pairs_all_random(self, input_dict, freq_dict, samples_path="samples", all_close=False, is_visual=False):
        object_ids = list(input_dict.keys())
        object_ids.sort()
        probs = [freq_dict[object_id] for object_id in object_ids]
        i = 0
        while (i < self.max_pairs):
            x1_mid, x2_mid = np.random.choice(
                object_ids, size=2, p=probs, replace=False)
            assert x1_mid != x2_mid
            is_close = np.random.rand() <= 0.5
            euclidean = np.random.rand() <= 0.5
            occx1 = matchesx1 = input_dict[x1_mid]
            occx2 = matchesx2 = input_dict[x2_mid]
            occx1 = list(occx1.keys())
            occx2 = list(occx2.keys())
            occx1.sort()
            occx2.sort()
            occx1 = np.array(occx1)
            occx2 = np.array(occx2)
            x1 = pivot = np.random.choice(occx1, size=1)[0]
            occx2 = occx2[occx2 != pivot]
            if (len(occx2) <= 1):
                continue
            x2 = np.random.choice(occx2, size=1)[0]
            x2_dict = matchesx2[x2]
            x2f = self.construct_sample_from_dict(matchesx2[x2])
            pair_name = f"{int(x1_mid)}d{int(x2_mid)}-{x1}-{x2}-0"
            x1f = self.construct_sample_from_dict(matchesx1[x1])
            pair = np.vstack((x1f, x2f))
            np.savetxt(
                f"{self.data_set_path}/{samples_path}/{pair_name}.txt", pair)
            i += 1

    def write_visual_folder(self, object_id, x1_frame, x2_frame, x1_dict, x2_dict, samples_path, is_positive=True, x2_id=None):
        folder_name = f"{int(object_id)}-{x1_frame}-{x2_frame}-1"
        if (not is_positive):
            folder_name = f"{int(object_id)}-{int(x2_id)}-{x1_frame}-{x2_frame}-0"
        # x1_frame =
        x1_img = self.get_img_crop_from_frame(x1_dict["bb"], x1_frame)
        x2_img = self.get_img_crop_from_frame(x2_dict["bb"], x2_frame)
        x1f = self.construct_sample_from_dict(x1_dict)
        x2f = self.construct_sample_from_dict(
            x2_dict)
        dir_path = f"{self.data_set_path}/{samples_path}/{folder_name}"
        os.makedirs(dir_path, exist_ok=True)
        x1_img.save(f"{dir_path}/x1.jpg")
        x2_img.save(f"{dir_path}/x2.jpg")
        np.savetxt(f"{dir_path}/x1.txt", x1f)
        np.savetxt(f"{dir_path}/x2.txt", x2f)
