from tabnanny import verbose
from lutils.general import get_img_crop_from_frame, get_img_crop_from_frame_no_padding
import os
import pandas as pd
from ultralytics import YOLO
from lutils.yollov11x_features import _predict_once, non_max_suppression, get_object_features
from types import MethodType
from ultralytics import YOLO, ASSETS
import cv2
from ultralytics.utils.ops import xywh2xyxy, scale_boxes
from ultralytics.engine.results import Results
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from deep_sort.Detection import Detection
from deep_sort import linear_assignment
from deep_sort import iou_matching


class WriteBenchmarkFeatures():

    def __init__(self,
                 bench_mark_path="../logs/benchmarks/clip_1/corrected",
                 raw_frames_path="../logs/benchmarks/clip_1/raw_frames",
                 ):
        self.bench_mark_path = bench_mark_path
        self.raw_frames_path = raw_frames_path
        self.labels_gt = f"{self.bench_mark_path}/labels"
        self.model = self.load_model()
        self.pose = YOLO(model="yolo11x-pose.pt", verbose=False)
        self.pose.fuse()
        self.averaged = 0

    def load_model(self):
        self.model = YOLO("yolo11x.pt")
        self.model.fuse()
        self.model.model._predict_once = MethodType(
            _predict_once, self.model.model)
        _ = self.model(ASSETS / "bus.jpg", save=False, embed=[16, 19, 22, 23])
        return self.model

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
        return result

    def convert_pil_bgr_to_rgb(pil_image):
        # Convert PIL image to NumPy array
        bgr_array = np.array(pil_image)

        # Convert BGR to RGB by reversing the last axis
        rgb_array = bgr_array[..., ::-1]

        # Convert back to a PIL image
        rgb_image = Image.fromarray(rgb_array, 'RGB')
        return rgb_image

    def construct_detections(self, outputs, is_pred=False):
        detections = []
        if outputs.ndim == 1 and len(outputs) > 0:
            row = outputs
            bb = row[:-1]
            id = int(row[-1])
            detection = Detection(torch.Tensor(bb))
            detection.time_since_update = 1
            detection.id = id

            detections.append(detection)
            return detections
        for row in outputs:
            bb = row[:-1]
            id = int(row[-1])
            if (id is None):
                import pdb
                pdb.set_trace()
            detection = Detection(torch.Tensor(bb))
            detection.time_since_update = 1
            detection.id = id
            detections.append(detection)
        return detections

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

    def get_detections_objects(self, det, frame):
        results = self.get_full_pred(det)
        features = det.feats
        objects = list(map(Detection, results))
        for i in range(len(features)):
            feat = features[i]
            detection = objects[i]
            detection.set_feature(feat)
        return objects

    def match_iou(self, gt, pred):
        matches, unmatched_gt, unmatched_pred = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, 0.7,
                gt, pred)

        return matches, unmatched_gt, unmatched_pred

    def adjust_ids(self, matches, gt, det):
        for match in matches:
            idx1, idx2 = match
            id = gt[idx1].id
            det[idx2].set_id(id)

    def get_bb_feaures(self, gt, img):
        results = self.predict(img)
        results = self.get_detections_objects(results, None)
        matches, unmatched_gt, unmatched_pred = self.match_iou(
            gt, results)
        self.adjust_ids(matches, gt, results)
        return results

    def write_features(self, detections, frame):
        # Construct the output path for the features
        write_path = f"{self.bench_mark_path}/featuresv2/{frame}"
        if len(detections) == 0:
            np.savetxt(write_path, [], delimiter=" ")
            # Ensure the directory exists
        os.makedirs(os.path.dirname(write_path), exist_ok=True)

        # Prepare data for saving
        feature_data = []
        for detection in detections:
            # Convert detection.id to an integer and concatenate with the features
            # Ensure id is an integer
            if (detection.id is not None):
                row = [int(detection.id)] + list(detection.feature)
                feature_data.append(row)

        # Convert to a NumPy array
        feature_array = np.array(feature_data)

        # Save to file using numpy.savetxt
        fmt = "%d " + " ".join(["%.6f"] * (feature_array.shape[1] - 1))
        np.savetxt(write_path, feature_array, fmt=fmt, delimiter=" ")

    def get_keypoints2(self, detections, img):
        feature_data = []
        crops = []
        ids = []
        for detection in detections:
            bbox = detection.xyxy
            crop = get_img_crop_from_frame_no_padding(bbox, img)
            crops.append(crop)
            result = self.pose(crop, verbose=False, embed=[22])
            keypoints = result[0].keypoints
            n = torch.clone(keypoints.xyn)
            keypoints = torch.clone(keypoints.data)

            keypoints[:, :, :2] = n[:, :, :2]
            # import pdb
            # pdb.set_trace()
            if (keypoints.shape[0] > 1):
                keypoints = torch.mean(keypoints, dim=0)
                self.averaged += 1

            keypoints = keypoints.flatten().cpu().numpy()
            if (len(keypoints) > 0 and detection.id is not None):
                row = [int(detection.id)] + list(keypoints)
                feature_data.append(row)
        return feature_data

    def get_keypoints(self, detections, img):
        feature_data = []
        crops = []
        ids = []
        for detection in detections:
            bbox = detection.xyxy
            crop = get_img_crop_from_frame_no_padding(bbox, img)
            crops.append(crop)
            ids.append(detection.id)

        result = self.pose(crops, verbose=False, embed=[22])
        result = torch.stack(result).cpu().tolist()
        # if (keypoints.shape[0] > 1):
        #         keypoints = torch.mean(keypoints, dim=0)
        #         self.averaged += 1

        #     keypoints = keypoints.flatten().cpu().numpy()
        #     if (len(keypoints) > 0 and detection.id is not None):
        #         row = [int(detection.id)] + list(keypoints)
        #         feature_data.append(row)
        for idx, id in enumerate(ids):
            if (id is None):
                print("NONE ID")
                continue
            row = [int(id)] + result[idx]
            feature_data.append(row)

        return feature_data

    def save_keypoints(self, keypoints, frame):
        write_path = f"{self.bench_mark_path}/keypointsv2/{frame}"
        if (len(keypoints) == 0):
            np.savetxt(write_path, keypoints, delimiter=" ")
            return

        keypoints = np.array(keypoints)
        # Ensure the directory exists
        os.makedirs(os.path.dirname(write_path), exist_ok=True)
        fmt = "%d " + " ".join(["%.6f"] * (keypoints.shape[1] - 1))
        np.savetxt(write_path, keypoints, fmt=fmt, delimiter=" ")

    def run(self):
        with torch.no_grad():
            raw_frames = os.listdir(self.raw_frames_path)
            for frame in tqdm(raw_frames, desc="preprocessing frames"):
                f_path = f"{self.raw_frames_path}/{frame}"
                f = frame.replace(".jpg", ".txt")
                gt = np.loadtxt(f"{self.labels_gt}/{f}")
                img = cv2.imread(f_path)
                gt_det = self.construct_detections(gt, None)
                # detections = self.get_bb_feaures(gt_det, img)
                # self.write_features(detections, f)
                keypoints = self.get_keypoints(gt_det, img)
                self.save_keypoints(keypoints, f)

            print("Total Averaged:", self.averaged)


if __name__ == "__main__":
    benchmark = WriteBenchmarkFeatures()
    benchmark.run()
    # write_video()
    # write_kp()
    # create()
    # check_dirs()
