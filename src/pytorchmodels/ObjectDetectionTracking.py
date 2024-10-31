from pytorchmodels.ObjectDetection import ObjectDetection
import cv2
import torch
import cvzone
from sort import *
from tqdm.notebook import tqdm
from pytorchmodels.Sort import Sort
from pytorchmodels.KalmanBoxTracker import KalmanBoxTracker


class ObjectDetectionTracking(ObjectDetection):

    def __init__(self, capture) -> None:
        super().__init__(capture)
        self.max_age = 100
        self.min_hits = 3
        self.iou_threshold = 0.3

    def process_video(self, video, write_path="./logs/outputLive/"):
        KalmanBoxTracker.count = 0
        cap = cv2.VideoCapture(video)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
       # total_frames = 10
        frame = 0
        mot_tracker = Sort(max_age=self.max_age,
                           min_hits=self.min_hits,
                           iou_threshold=self.iou_threshold)
        with tqdm(total=total_frames-1, desc="Processing frames", unit="frame") as pbar:
            while True:
                _, img = cap.read()
                det = self.predict(img)
                detections = self.get_full_pred(det)

                results = mot_tracker.update(detections)
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
                    img, f'{current_class[0]}, {id}', (x1, y1), scale=1, thickness=1, colorR=(0, 0, 255))

                cvzone.cornerRect(img, (x1, y1, w, h), l=9,
                                  rt=1, colorR=(255, 0, 255))
        return img
