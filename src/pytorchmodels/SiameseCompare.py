import numpy as np
from ultralytics import YOLO, ASSETS
from types import MethodType
from lutils.yollov11_features import _predict_once, non_max_suppression, get_object_features
from ultralytics.utils.ops import xywh2xyxy, scale_boxes
import torch
from ultralytics.engine.results import Results
from deep_sort.Detection import Detection
from lutils.general import write_json, get_img_crop_from_frame
from torchvision import transforms
import cv2
import cvzone
from PIL import Image
from deep_sort.sk_learn_linear_assignment import linear_assignment

visual_siamese = "../logs/training/siamese-resnet/visual_samples_100k/2024_11_14_03_02_33/checkpoints/epoch-best.ckpt"

transform = transforms.Compose([
    # Resize to 224x224 (or your target size)
    transforms.Resize((224, 224)),
    transforms.ToTensor(),          # Convert image to tensor
    # Normalize for pre-trained models
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
        0.229, 0.224, 0.225])
])


class SiamaeseCompare():

    def __init__(self):
        self.yolo = self.load_model()

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
        # import pdb
        # pdb.set_trace()
        return result

    def get_visual_crop_features(self, det, frame):
        boxes = det.boxes.xyxy
        features = []
        for box in boxes:
            img_crop = get_img_crop_from_frame(box, frame).convert('RGB')
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

    def get_detections_objects(self, det, frame):
        results = self.get_full_pred(det)
        features = det.feats
        objects = list(map(Detection, results))
        # if (self.use_siamese):
        #     if (self.use_visual_siamese):
        #         features = self.get_visual_crop_features(det, frame)
        #     else:
        #         with torch.no_grad():
        #             if len(features) > 0:
        #                 if (len(features.shape) == 1):
        #                     features = features.unsqueeze(dim=0)
        #                 try:
        #                     self.siamese_network.eval()
        #                     features = self.siamese_network.network.backbone(
        #                         features)
        #                 except:
        #                     import pdb
        #                     pdb.set_trace()

        if (len(features.shape) == 1):
            features = features.unsqueeze(dim=0)
        try:
            for i in range(len(features)):
                feat = features[i]
                detection = objects[i]
                detection.set_feature(feat)
        except:
            import pdb
            pdb.set_trace()
        return objects

    def process_pair_frames(self, f1, f2):
        d_a, img_a = self.process_frame(f1)
        d_b, img_b = self.process_frame(f2)
        cost_matrix = self.get_cost_matrix((d_a), d_b)
        matches, unmatched_a, unmatched_b = self.min_cost_matching(
            cost_matrix, d_a, d_b)
        d_a, d_b = self.adjust_ids(d_a, d_b, matches, unmatched_a, unmatched_b)
        img_a, img_b = self.plot_boxes(
            (d_a), img_a), self.plot_boxes(d_b, img_b)
        return {
            "cost_matrix": cost_matrix, "matches": matches, "u_a": unmatched_a, "u_b": unmatched_b,
            "img_a": img_a, "img_b": img_b
        }

    def process_frame(self, img):
        img = Image.open(img).convert("RGB")
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        det = self.predict((img))
        detections = self.get_detections_objects(det, img)
        return detections, img

    def get_cost_matrix(self, d_a, d_b):
        f_a = [d.feature for d in d_a]
        f_b = [d.feature for d in d_b]
        cost_matrix = torch.zeros((len(d_a), len(d_b)))
        for i, target in enumerate(f_a):
            cost_matrix[i, :] = 1 - \
                torch.nn.functional.cosine_similarity(
                    torch.Tensor(target), torch.Tensor(f_b))
        return cost_matrix

    # def gate_cost_neural(cost_matrix, threshhold):
    # cost_matrix[cost_matrix > threshhold] = INFTY_COST
    # return cost_matrix

    def min_cost_matching(self,
                          cost_matrix, tracks, detections, max_distance=1e5):

        track_indices = np.arange(len(tracks))
        detection_indices = np.arange(len(detections))

        if len(detection_indices) == 0 or len(track_indices) == 0:
            return [], track_indices, detection_indices  # Nothing to match.

        cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5
        indices = linear_assignment(cost_matrix)

        matches, unmatched_tracks, unmatched_detections = [], [], []
        for col, detection_idx in enumerate(detection_indices):
            if col not in indices[:, 1]:
                unmatched_detections.append(detection_idx)
        for row, track_idx in enumerate(track_indices):
            if row not in indices[:, 0]:
                unmatched_tracks.append(track_idx)
        for row, col in indices:
            track_idx = track_indices[row]
            detection_idx = detection_indices[col]
            if cost_matrix[row, col] > max_distance:
                unmatched_tracks.append(track_idx)
                unmatched_detections.append(detection_idx)
            else:
                matches.append((track_idx, detection_idx))
        return matches, unmatched_tracks, unmatched_detections

    def adjust_ids(self, d_a, d_b, matches, unmatched_a, unmatched_b):
        for i in range(len(d_a)):
            d_a[i].id = i

        for j in range(len(d_b)):
            d_b[j].id = self.find_match(matches, j)
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

            if (conf > 0.25):
                # cvzone.putTextRect(
                #     img, f'{id}, {conf}', (x1, y1), scale=1, thickness=1, colorR=(0, 0, 255))
                cvzone.putTextRect(
                    img, f'{id}', (x1, y1-5), scale=1, thickness=1, colorR=(0, 0, 255))
                cvzone.cornerRect(img, (x1, y1, w, h), l=9,
                                  rt=1, colorR=(255, 0, 255))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
