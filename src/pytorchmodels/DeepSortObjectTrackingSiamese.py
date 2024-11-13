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
from lutils.general import write_json
from ultralytics.utils.ops import xywh2xyxy, scale_boxes
from ultralytics.engine.results import Results
import copy
from pytorchmodels.Siamese import Siamese
import yaml

"logs/training/siamese-lin-bn-conv/full-ds-2/2024_11_12_19_50_03/checkpoints/epoch-best.ckpt"
nn_margin = 0.1


class DeepSortObjectTrackingSiamese(ObjectDetection):

    def __init__(self, capture, write_path, use_kalman=True) -> None:
        super().__init__(capture)
        self.min_confidence = 0.25
        self.max_cosine_distance = 0.2
        self.nn_budget = None
        self.write_path = write_path
        self.frame_count = 0
        self.siamese_ckpt_path = "../logs/training/siamese-lin-bn-conv/full-ds-2/2024_11_12_19_50_03/checkpoints/epoch-best.ckpt"
        self.configs_path = "./configs/global_configs.yaml"
        self.use_kalman = use_kalman

        with open(self.configs_path, "r") as in_file:
            self.global_configs = yaml.safe_load(in_file)
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
        self.siamese_network.load_ckpt(self.siamese_ckpt_path)
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

        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", self.max_cosine_distance, None)
        tracker = Tracker(
            metric, print_cost_matrix=print_cost_matrix, max_age=max_age)
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

    def get_detections_objects(self, det, frame):
        results = self.get_full_pred(det)
        features = det.feats
        objects = list(map(Detection, results))
        with torch.no_grad():
            features = self.siamese_network.network.backbone(features)
        # import pdb
        # pdb.set_trace()
        for i in range(len(features)):
            feat = features[i]
            detection = objects[i]
            detection.set_feature(feat)
        return objects

    def get_crops(self, res, frame):
        crop_objects = []
        for box in res:
            crop_obj = frame[int(box[1]): int(
                box[3]), int(box[0]): int(box[2])]
            crop_objects.append(crop_obj)
        return crop_objects

    def get_full_pred(self, det):
        try:
            boxes = det.boxes.xyxy
        except:
            import pdb
            pdb.set_trace()

        cls = det.boxes.cls.unsqueeze(1)
        conf = det.boxes.conf.unsqueeze(1)
        detections = torch.cat((boxes, conf, cls), dim=1)
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
