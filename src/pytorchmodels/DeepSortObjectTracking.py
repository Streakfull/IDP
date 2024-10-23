from pytorchmodels.ObjectDetection import ObjectDetection
import cv2
import torch
import cvzone
from sort import *
from tqdm.notebook import tqdm
from deep_sort.Detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import nn_matching
import numpy as np


class DeepSortObjectTracking(ObjectDetection):

    def __init__(self, capture) -> None:
        super().__init__(capture)
        self.min_confidence = 0.25
        self.max_cosine_distance = 0.2
        self.nn_budget = 0.2

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
                detections = self.get_detections_objects(det)
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

                    cls = cls.numpy()
                    cls = np.array([cls])
                    conf = np.array([conf])
                    id = np.array([track.track_id])
                    result = np.concatenate((bbox, conf, cls, id))
                    results.append(result)
                frames = self.plot_boxes(results, img)

                cv2.imwrite(f"{write_path}/frame_{frame}.jpg", frames)
                frame += 1
                pbar.update(1)
                if (cv2.waitKey(1) == ord('q')):
                    break
                if frame == total_frames:
                    break
            cap.release()
            cv2.destroyAllWindows()

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

    def get_detections_objects(self, det):
        results = self.get_full_pred(det)
        objects = list(map(Detection, results))
        return objects
