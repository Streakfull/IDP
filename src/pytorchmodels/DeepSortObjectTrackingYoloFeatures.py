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
from ultralytics.utils.ops import xywh2xyxy, scale_boxes
from ultralytics.engine.results import Results


class DeepSortObjectTrackingYoloFeatures(ObjectDetection):

    def __init__(self, capture, write_path) -> None:
        super().__init__(capture)
        self.min_confidence = 0.25
        self.max_cosine_distance = 0.2
        self.nn_budget = None
        self.write_path = write_path
        self.frame_count = 0

    def load_model(self):
        model = super().load_model()
        self.model = model
        self.model.model._predict_once = MethodType(_predict_once, model.model)
        _ = self.model(ASSETS / "bus.jpg", save=False, embed=[16, 19, 22, 23])
        return model

    def predict(self, img):
        prepped = self.model.predictor.preprocess([img])
        result = self.model.predictor.inference(prepped)
        output, idxs = non_max_suppression(result[-1][0], in_place=False)
        obj_feats = get_object_features(result[:3], idxs[0].tolist())
        output[0][:, :4] = scale_boxes(
            prepped.shape[2:], output[0][:, :4], img.shape)
        result = Results(
            img, path="", names=self.model.predictor.model.names, boxes=output[0])
        result.feats = obj_feats

        return result

    def process_video(self, video, write_path="./logs/outputLive/"):
        cap = cv2.VideoCapture(video)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # total_frames = 10
        frame = 0
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", self.max_cosine_distance, None)
        tracker = Tracker(metric)
       # results = []
        with tqdm(total=total_frames-1, desc="Processing frames", unit="frame") as pbar:
            while True:
                _, img = cap.read()
                det = self.predict(img)

                detections = self.get_detections_objects(det, img)

                detections = [
                    d for d in detections if d.confidence >= self.min_confidence]

                tracker.predict()
                tracker.update(detections)
                results = []
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
                    results.append(result)
                frames = self.plot_boxes(results, img)

                cv2.imwrite(f"{write_path}/frame_{frame}.jpg", frames)
                frame += 1
                self.frame_count += 1
                pbar.update(1)
                if (cv2.waitKey(1) == ord('q')):
                    break
                if frame == total_frames:
                    break
            cap.release()
            cv2.destroyAllWindows()
        return tracker.metrics

    def plot_boxes(self, results, img):
        for box in results:

            x1, y1, x2, y2, conf, cls, id = box
            x1, y1, x2, y2, conf, cls, id = int(x1), int(y1), int(
                x2), int(y2), round(float(conf), 2), int(cls), int(id)
            w, h = x2-x1, y2-y1
            current_class = self.CLASS_NAMES_DICT[cls]

            if (conf > 0.25):
                cvzone.putTextRect(
                    img, f'{current_class[0]}, {id}, {conf}', (x1, y1), scale=1, thickness=1, colorR=(0, 0, 255))

                cvzone.cornerRect(img, (x1, y1, w, h), l=9,
                                  rt=1, colorR=(255, 0, 255))
        return img

    def get_detections_objects(self, det, frame):
        results = self.get_full_pred(det)
        features = det.feats
        objects = list(map(Detection, results))
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
