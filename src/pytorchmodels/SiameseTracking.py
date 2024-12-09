from tqdm import tqdm
import numpy as np
from ultralytics import YOLO, ASSETS
from types import MethodType
from lutils.yollov11_features import _predict_once, non_max_suppression, get_object_features
from ultralytics.utils.ops import xywh2xyxy, scale_boxes
import torch
from ultralytics.engine.results import Results
from deep_sort.Detection import Detection
from lutils.general import find_object_by_id, write_json, get_img_crop_from_frame, extract_frame_number
from torchvision import transforms
from deep_sort import nn_matching
import cv2
import cvzone
from PIL import Image
from deep_sort.sk_learn_linear_assignment import linear_assignment
from pytorchmodels.ObjectDetection import ObjectDetection
from pytorchmodels.Siamese import Siamese
from training import ModelBuilder
import yaml
from deep_sort import iou_matching
from deep_sort.linear_assignment import min_cost_matching
import os
from siamese_tracking.tracker import Tracker


transform = transforms.Compose([
    # Resize to 224x224 (or your target size)
    transforms.Resize((224, 224)),
    transforms.ToTensor(),          # Convert image to tensor
    # Normalize for pre-trained models
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
        0.229, 0.224, 0.225])
])


class SiameseTracking(ObjectDetection):
    def __init__(self, capture, write_path, use_enhanced_tracking=False, track_queue_size=10) -> None:
        super().__init__(capture)
        self.min_confidence = 0.25
        self.max_cosine_distance = 0.2
        self.write_path = write_path
        self.frame_count = 0
        self.use_enhanced_tracking = use_enhanced_tracking
        self.track_queue_size = track_queue_size
        with open(self.configs_path, "r") as in_file:
            self.global_configs = yaml.safe_load(in_file)
            self.siamese_configs = self.global_configs["model"]["siamese"]
            self.siamese_network = self.load_siamese()

    def load_siamese(self):
        self.siamese_network = Siamese(self.siamese_configs)
        self.siamese_network.load_ckpt(
            self.global_configs["training"]["ckpt_path"])
        device = "cuda:0"
        self.siamese_network.to(device)
        return self.siamese_network

    def load_model(self):
        model = YOLO("yolo11n.pt")
        self.configs_path = "./configs/global_configs.yaml"
        model.fuse()
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
        indices = torch.nonzero(output[0][:, -1] <= 0).squeeze()
        boxes = output[0][indices]
        obj_feats = obj_feats[indices]
        result = Results(
            img, path="", names=self.model.predictor.model.names, boxes=boxes)
        result.feats = obj_feats
        return result

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
        for idx, box in enumerate(boxes):
            img_crop = get_img_crop_from_frame(box, frame).convert('RGB')
            img_crop = self.convert_pil_bgr_to_rgb(img_crop)

            # img_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
            save_path = os.path.join(
                self.write_path, "crops", f"crop_{idx}.jpg")
            # img_crop.save(save_path)
            img_crop = transform(img_crop)
            with torch.no_grad():
                self.siamese_network.network.resnet.eval()
                self.siamese_network.eval()
                self.siamese_network.network.eval()
                feat = self.siamese_network.network.img_feature(
                    img_crop.to("cuda:0").unsqueeze(0))
                features.append(feat.squeeze(0))
        return features

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

    def adjust_ids(self, d_a, d_b, matches, unmatched_a, unmatched_b, c_matrix):

        for i in range(len(d_a)):
            d_a[i].id = i

        for j in range(len(d_b)):
            id = self.find_match(matches, j)
            d_b[j].id = id
            if c_matrix[id, j] == 1e5:
                d_b[j].id = -1
        return d_a, d_b

    def find_match(self, pairs, target):
        for first, second in pairs:
            if second == target:
                return first
        return -1

    def plot_boxes(self, detections, img):
        for detection in detections:
            x1, y1, x2, y2 = detection.to_tlbr()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            id = detection.id
            conf = detection.confidence
            w, h = x2-x1, y2-y1
            # cvzone.putTextRect(
            #     img, f'{id}, {conf}', (x1, y1), scale=1, thickness=1, colorR=(0, 0, 255))
            cvzone.putTextRect(
                img, f'{id}', (x1, y1-5), scale=1, thickness=1, colorR=(0, 0, 255))
            cvzone.cornerRect(img, (x1, y1, w, h), l=9,
                              rt=1, colorR=(255, 0, 255))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def get_detections_objects(self, det, frame):
        results = self.get_full_pred(det)
        features = det.feats
        objects = list(map(Detection, results))
        is_img_features = self.siamese_configs["use_visual"] or self.siamese_configs["use_combined"]
        is_bb = self.siamese_configs["use_combined"] or self.siamese_configs["use_bb"]
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

    def process_video(self, video,
                      write_path="./logs/outputLive/",
                      labels_write_path=None,
                      max_frames=None,
                      start_frame=None,
                      write_directly=True,
                      max_age=3
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
            metric=metric, use_enhance=self.use_enhanced_tracking, track_queue_size=self.track_queue_size)

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
                pred_det = tracker.update(detections, self.frame_count)
                if self.use_enhanced_tracking:
                    if (len(pred_det) > 0):
                        print("KF Detection: ", len(pred_det))
                    detections = detections + pred_det

                frames = self.plot_boxes(detections=detections, img=img)
                frame_name = f"frame_{frame}"
                results = []
                tracks_feat = []
                plotted_frames = []
                frame_labels = []
                features = []
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
                frame_labels.append(results)
                features.append(tracks_feat)

                if (labels_write_path is not None and len(results) > 0):
                    fmt = ['%.4f'] * (results[0].shape[0] - 1) +\
                        ['%d'] if len(results) > 0 else '%d'
                    np.savetxt(f"{labels_write_path}/{frame_name}.txt",
                               results, delimiter=' ', fmt=fmt)
                if write_directly:
                    f = cv2.imwrite(f"{write_path}/{frame_name}.jpg", frames)
                else:
                    plotted_frames.append(frames)
                frame += 1
                self.frame_count += 1
                pbar.update(1)
                if (cv2.waitKey(1) == ord('q')):
                    break
                if frame >= total_frames:
                    break
            cap.release()
            cv2.destroyAllWindows()
        return plotted_frames, frame_labels, features
