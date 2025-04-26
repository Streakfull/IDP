from types import MethodType

from ultralytics import YOLO, ASSETS
from pytorchmodels.ObjectDetection import ObjectDetection
import cv2
import torch
import cvzone
from sort import *
from tqdm.notebook import tqdm
from deep_sort.Detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import nn_matching
from deep_sort.nms import non_max_suppression
import numpy as np
import uuid
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from PIL import Image
from lutils.yollov11_features import _predict_once, non_max_suppression, get_object_features
from lutils.general import write_json, get_img_crop_from_frame
from ultralytics.utils.ops import xywh2xyxy, scale_boxes
from ultralytics.engine.results import Results
import copy
from pytorchmodels.Siamese import Siamese
import yaml
import os

k_200 = "../logs/training/siamese-lin-bn-conv/200k-samples/2024_11_13_02_07_53/checkpoints/epoch-best.ckpt"
k_30 = "../logs/training/siamese-lin-bn-conv/full-ds-2/2024_11_12_19_50_03/checkpoints/epoch-best.ckpt"
close = "../logs/training/siamese-lin-bn-conv/close-samples/2024_11_13_23_11_48/checkpoints/epoch-best.ckpt"
negative_close = "../logs/training/siamese-lin-bn-conv/negative-close-samples/2024_11_14_00_11_50/checkpoints/epoch-best.ckpt"
visual_siamese = "../logs/training/siamese-resnet/visual_samples_100k/2024_11_14_03_02_33/checkpoints/epoch-best.ckpt"
random_200k = "../logs/training/siamese-lin-bn-conv/uniform_samples_200k/2024_11_14_04_21_24/checkpoints/epoch-best.ckpt"
text_10k = '../logs/training/eval/fc/10k/exp_1/2024_11_14_12_35_01/checkpoints/epoch-16.ckpt'
text_10k_uniform = "../logs/training/eval/fc/10k/uniform/2024_11_14_12_40_23/checkpoints/epoch-18.ckpt"
visual_10k = "../logs/training/eval/visual/10k/exp_1/2024_11_14_13_42_52/checkpoints/epoch-8.ckpt"
visual_30k = "../logs/training/eval/visual/30k/exp_1/2024_11_14_14_02_57/checkpoints/epoch-5.ckpt"
text_50k = '../logs/training/eval/fc/50k/exp_1/2024_11_14_12_46_46/checkpoints/epoch-14.ckpt'
text_100k = '../logs/training/eval/fc/100k/exp_1/2024_11_14_13_08_48/checkpoints/epoch-8.ckpt'
visual_100k = "../logs/training/eval/visual/100k/exp_1/2024_11_14_15_56_17/checkpoints/epoch-2.ckpt"
visual_200k = "../logs/training/eval/visual/200k/exp_1/2024_11_14_17_15_36/checkpoints/epoch-latest.ckpt"

transform = transforms.Compose([
    # Resize to 224x224 (or your target size)
    transforms.Resize((224, 224)),
    transforms.ToTensor(),          # Convert image to tensor
    # Normalize for pre-trained models
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
        0.229, 0.224, 0.225])
])


class DeepSortObjectTrackingSiamese(ObjectDetection):

    def __init__(self, capture, write_path, use_kalman=True, start_confirmed=False, use_visual_siamese=False, use_siamese=True) -> None:
        super().__init__(capture)
        self.min_confidence = 0.25
        self.max_cosine_distance = 0.2
        self.nn_budget = None
        self.write_path = write_path
        self.frame_count = 0
        # self.siamese_ckpt_path = random_200k
        self.configs_path = "./configs/global_configs.yaml"
        self.use_kalman = use_kalman
        self.start_confirmed = start_confirmed
        self.use_visual_siamese = use_visual_siamese
        self.use_siamese = use_siamese

        with open(self.configs_path, "r") as in_file:
            self.global_configs = yaml.safe_load(in_file)
            self.siamese_configs = self.global_configs["model"]["siamese"]
        self.siamese_network = self.load_siamese()

    def load_model(self):
        model = super().load_model()
        self.model = model
        self.model.model._predict_once = MethodType(_predict_once, model.model)
        _ = self.model(ASSETS / "bus.jpg", save=False, embed=[16, 19, 22, 23])
        return model

    def load_siamese(self):
        configs = self.global_configs["model"]["siamese"]
        self.siamese_network = Siamese(configs)
        ckpt_path = self.global_configs["training"]["ckpt_path"]
        self.siamese_network.load_ckpt(ckpt_path=ckpt_path)
        device = "cuda:0"
        self.siamese_network.to(device)
        return self.siamese_network

    def predict(self, img):
        prepped = self.model.predictor.preprocess([img])
        result = self.model.predictor.inference(prepped)
        output, idxs = non_max_suppression(result[-1][0], in_place=False)
        obj_feats = get_object_features(result[:3], idxs[0].tolist())
        output[0][:, :4] = scale_boxes(
            prepped.shape[2:], output[0][:, :4], img.shape)
        indices = torch.nonzero(output[0][:, -1] <= 0).squeeze()
        boxes = output[0][indices]
        obj_feats = obj_feats[indices]
        result = Results(
            img, path="", names=self.model.predictor.model.names, boxes=boxes)
        result.feats = obj_feats
        # import pdb
        # pdb.set_trace()
        return result

    def plot_benchmark(self, video, write_path=None, labels_path=None, max_frames=None, start_frame=None):
        cap = cv2.VideoCapture(video)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if (max_frames is not None):
            total_frames = max_frames
        frame = 0
        if (start_frame is not None):
            if (max_frames is not None):
                total_frames += start_frame
        with tqdm(total=total_frames-1, desc="Processing frames", unit="frame") as pbar:
            while True:
                _, img = cap.read()
                if (start_frame is not None and frame < start_frame):
                    frame += 1
                    pbar.update(1)
                    continue

                labels = np.loadtxt(f"{labels_path}/frame_{frame}.txt")
                frames = self.plot_boxes(labels, img)
                cv2.imwrite(f"{write_path}/frame_{frame}.jpg", frames)
                frame += 1
                pbar.update(1)
                if frame >= total_frames:
                    cap.release()
                    cv2.destroyAllWindows()
                    break

    def process_video(self, video,
                      write_path="./logs/outputLive/",
                      labels_write_path=None, max_frames=None,
                      frame_difference_write_path=None, start_frame=None,
                      write_directly=True,
                      print_cost_matrix=True,
                      max_age=30


                      ):
        cap = cv2.VideoCapture(video)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if (max_frames is not None):
            total_frames = max_frames
        frame = 0
        self.frame_count = 0
        if (start_frame is not None):
            if (max_frames is not None):
                total_frames += start_frame

        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", self.max_cosine_distance, None)
        tracker = Tracker(
            metric,
            print_cost_matrix=print_cost_matrix,
            max_age=max_age,
            use_kalman=self.use_kalman,
            start_confirmed=self.start_confirmed)

        stored_metrics = copy.deepcopy(tracker.metrics)
        all_frames_diff = {}
        plotted_frames = []
        frame_labels = []
        features = []
        with tqdm(total=total_frames-1, desc="Processing frames", unit="frame") as pbar:
            while True:
                _, img = cap.read()
                if (start_frame is not None and frame < start_frame):
                    frame += 1
                    self.frame_count += 1
                    pbar.update(1)
                    continue

                det = self.predict(img)

                detections = self.get_detections_objects(det, img)

                detections = [
                    d for d in detections if d.confidence >= self.min_confidence]

                tracker.predict()
                tracker.update(detections)

                results = []
                tracks_feat = []
                for track in tracker.tracks:
                    if (not track.is_confirmed() or track.time_since_update > 1) and (frame > tracker.n_init):
                        continue

                    bbox = track.to_tlbr()
                   # bbox = track.get_detection().to_tlbr()
                    conf = track.get_detection().get_conf()
                    cls = track.get_detection().get_cls()

                    cls = cls.cpu().numpy()
                    cls = np.array([cls])
                    conf = np.array([conf])
                    id = np.array([track.track_id])
                    result = np.concatenate((bbox, conf, cls, id))
                    tracks_feat.append(
                        (int(id), track.get_detection().get_feature()))
                    results.append(result)
                frames = self.plot_boxes(results, img)
                frame_name = f"frame_{frame}"
                frame_labels.append(results)
                features.append(tracks_feat)
                if (labels_write_path is not None):
                    fmt = ['%.4f'] * (result.shape[0] - 1) +\
                        ['%d'] if len(results) > 0 else '%d'
                    np.savetxt(f"{labels_write_path}/{frame_name}.txt",
                               results, delimiter=' ', fmt=fmt)

                    # frame_diff = self.get_frame_diff(
                    #     stored_metrics, tracker.metrics)
                    # if (frame_diff is not None):
                    #     all_frames_diff[frame] = frame_diff
                    #     write_json(
                    #         all_frames_diff,  f"{frame_difference_write_path}/frame_diff.json")
                if write_directly:
                    cv2.imwrite(f"{write_path}/{frame_name}.jpg", frames)
                   # cv2.imwrite(f"{write_path}/{frame_name}.jpg", img)
                else:
                    plotted_frames.append(frames)
                frame += 1
                self.frame_count += 1
                stored_metrics = copy.deepcopy(tracker.metrics)
                pbar.update(1)
                if (cv2.waitKey(1) == ord('q')):
                    break
                if frame >= total_frames:
                    break
            cap.release()
            cv2.destroyAllWindows()
        return plotted_frames, frame_labels, features
        # return tracker.metrics

    def plot_boxes(self, results, img):
        for box in results:

            x1, y1, x2, y2, conf, cls, id = box
            x1, y1, x2, y2, conf, cls, id = int(x1), int(y1), int(
                x2), int(y2), round(float(conf), 2), int(cls), int(id)
            w, h = x2-x1, y2-y1
            current_class = self.CLASS_NAMES_DICT[cls]

            if (conf > 0.25):
                # cvzone.putTextRect(
                #     img, f'{id}, {conf}', (x1, y1), scale=1, thickness=1, colorR=(0, 0, 255))
                cvzone.putTextRect(
                    img, f'{id}', (x1, y1-5), scale=1, thickness=1, colorR=(0, 0, 255))
                cvzone.cornerRect(img, (x1, y1, w, h), l=9,
                                  rt=1, colorR=(255, 0, 255))
        return img

    # def get_detections_objects(self, det, frame):
    #     results = self.get_full_pred(det)
    #     features = det.feats
    #     objects = list(map(Detection, results))
    #     if (self.use_siamese):
    #         if (self.use_visual_siamese):
    #             features = self.get_visual_crop_features(det, frame)
    #         else:
    #             with torch.no_grad():
    #                 if len(features) > 0:
    #                     if (len(features.shape) == 1):
    #                         features = features.unsqueeze(dim=0)
    #                     try:
    #                         self.siamese_network.eval()
    #                         features = self.siamese_network.network.backbone(
    #                             features)
    #                     except:
    #                         import pdb
    #                         pdb.set_trace()

    #     if (len(features.shape) == 1):
    #         features = features.unsqueeze(dim=0)
    #     try:
    #         for i in range(len(features)):
    #             feat = features[i]
    #             detection = objects[i]
    #             detection.set_feature(feat)
    #     except:
    #         import pdb
    #         pdb.set_trace()
    #     return objects

    def get_detections_objects(self, det, frame):
        results = self.get_full_pred(det)
        features = det.feats
        objects = list(map(Detection, results))
        is_img_features = self.siamese_configs["use_visual"] or self.siamese_configs["use_combined"]
        is_bb = self.siamese_configs["use_combined"] or self.siamese_configs["use_bb"]
        if not self.use_siamese:
            features = torch.ones_like(features)
        else:
            with torch.no_grad():
                self.siamese_network.eval()
                self.siamese_network.network.eval()
                if is_img_features:
                    img_features = self.get_visual_crop_features(det, frame)
                if is_bb:
                    bb_features = self.siamese_network.network.bb_features(
                        features)

            if (len(features.shape) == 1):
                features = features.unsqueeze(dim=0)
                if is_bb:
                    bb_features = bb_features.unsqueeze(dim=0)
                if is_img_features:
                    img_features = img_features.unsqueeze(dim=0)
        # try:
        for i in range(len(features)):
            if not self.use_siamese:
                feat = features[i]
            if is_bb:
                f = bb_features[i]
                feat = f
            if is_img_features:
                img = img_features[i]
                feat = img
            if is_bb and is_img_features:
                feat = torch.cat((img, f))

            detection = objects[i]
            detection.set_feature(feat)
        return objects

    def convert_pil_bgr_to_rgb(self, pil_image):
        # Convert PIL image to NumPy array
        bgr_array = np.array(pil_image)

        # Convert BGR to RGB by reversing the last axis
        rgb_array = bgr_array[..., ::-1]

        # Convert back to a PIL image
        rgb_image = Image.fromarray(rgb_array, 'RGB')
        return rgb_image

    def get_visual_crop_features(self, det, frame):
        boxes = det.boxes.xyxy
        features = []
        for box in boxes:
            img_crop = get_img_crop_from_frame(box, frame).convert('RGB')
            img_crop = self.convert_pil_bgr_to_rgb(img_crop)
            img_crop = transform(img_crop)
            with torch.no_grad():
                self.siamese_network.network.resnet.eval()
                self.siamese_network.eval()
                self.siamese_network.network.eval()
                feat = self.siamese_network.network.img_feature(
                    img_crop.to("cuda:0").unsqueeze(0))
                features.append(feat.squeeze(0))

        return torch.stack(features, dim=0)

    def get_full_pred(self, det):
        try:
            boxes = det.boxes.xyxy
        except:
            import pdb
            pdb.set_trace()

        cls = det.boxes.cls.unsqueeze(1)
        conf = det.boxes.conf.unsqueeze(1)
        detections = torch.cat((boxes, conf, cls), dim=1).cpu()
        return detections

    def get_frame_diff(self, stored_metrics, new_metrics):
        # Removed
        some_removed = False
        some_added = False
        removed = []
        added_ids = []
        if (stored_metrics['removed'] != new_metrics['removed']):
            some_removed = True
            new_ids = new_metrics['id_frames'].keys()
            for new_id in new_ids:
                if new_id in stored_metrics['id_frames'] and stored_metrics['id_frames'][new_id] == new_metrics['id_frames'][new_id]:
                    removed.append(new_id)

        if (stored_metrics['new_ids'] != new_metrics['new_ids']):
            some_added = True
            new_ids = new_metrics['id_frames'].keys()
            for new_id in new_ids:
                if new_id not in stored_metrics['id_frames']:
                    added_ids.append(new_id)

        if (some_removed or some_added):
            return {
                "removed": removed,
                "new_ids": added_ids
            }

        return None

    def process_pairwise_frames(self, ds,
                                write_path,
                                max_length=None):

        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", self.max_cosine_distance, None)
        total_ds = len(ds)
        frame_labels = []
        current_idx = 0
        if (max_length is not None):
            total_ds = max_length
        with tqdm(total=total_ds-1, desc="Processing ds", unit="data") as pbar:
            for i in range(total_ds):
                e = ds[current_idx]
                f1, f2, x1, x2 = e["f1"], e["f2"], e["x1"], e["x2"]
                tup = [(f1, x1), (f2, x2)]
                tracker = Tracker(
                    metric,
                    print_cost_matrix=False,
                    use_kalman=self.use_kalman,
                    start_confirmed=self.start_confirmed,
                    max_age=1)
                for j in range(len(tup)):
                    fr, img = tup[j]
                    det = self.predict(img)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                    detections = self.get_detections_objects(det, img)

                    # detections = [
                    #     d for d in detections if d.confidence >= self.min_confidence]
                    results = []
                    features = []
                    tracker.predict()
                    tracker.update(detections)
                    for detection in detections:
                        bbox = detection.to_tlbr()
                        conf = detection.get_conf()
                        cls = detection.get_cls()
                        cls = cls.cpu().numpy()
                        cls = np.array([cls])
                        conf = np.array([conf])
                        id = np.array([detection.id])
                        result = np.concatenate((bbox, conf, cls, id))
                        results.append(result)
                    frames = self.plot_boxes(results, img)
                    frame_name = f"frame_{fr}"
                    frame_labels.append(results)
                    # features.append(tracks_feat)
                    fmt = ['%.4f'] * (result.shape[0] - 1) +\
                        ['%d'] if len(results) > 0 else '%d'
                    dir_path = f"{write_path}/{f1}_{f2}"
                    os.makedirs(dir_path, exist_ok=True)
                    np.savetxt(f"{dir_path}/{fr}.txt",
                               results, delimiter=' ', fmt=fmt)
                    cv2.imwrite(f"{dir_path}/{fr}.jpg",
                                frames)

                pbar.update(1)
                if (current_idx >= total_ds):
                    break
                current_idx += 1
