import os
import cvzone
from pytorchmodels.ObjectDetection import ObjectDetection
from pytorchmodels.DeepIOUTracker.make_parser import make_parser
import yaml
from pytorchmodels.Siamese import Siamese
from datasets.SoccerNetMatches import SoccerNetDataset
from ultralytics import YOLO, ASSETS
from types import MethodType
from lutils.yollov11_features import _predict_once, non_max_suppression, get_object_features
from ultralytics.utils.ops import xywh2xyxy, scale_boxes
from ultralytics.engine.results import Results
from tqdm import tqdm
import numpy as np
import torch
import cv2
from deep_eiou.tracker import Deep_EIoU
from siamese_tracking.Detection import Detection


class DeepIOUTracker(ObjectDetection):
    def __init__(self, write_path) -> None:
        super().__init__("")
        self.args = make_parser().parse_args()
        with open(self.configs_path, "r") as in_file:
            self.global_configs = yaml.safe_load(in_file)
            self.siamese_configs = self.global_configs["model"]["siamese"]
            self.siamese_network = self.load_siamese()
        self.soccernet = SoccerNetDataset(
            "./raw_dataset/soccernet-tracking/raw/tracking/train2.txt", train=False)

    def load_siamese(self):
        self.siamese_network = Siamese(self.siamese_configs)
        self.siamese_network.load_ckpt(
            self.global_configs["training"]["ckpt_path"])
        device = "cuda:0"
        self.siamese_network.to(device)
        return self.siamese_network

    def load_model(self):
        model = YOLO("yolo11x.pt", verbose=False)
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

    def plot_boxes(self, tlwhs, ids, img):
        for tlwh, id in zip(tlwhs, ids):
            x1, y1, w, h = map(int, tlwh)  # Convert to integers

            # Draw bounding box using cvzone
            cvzone.cornerRect(img, (x1, y1, w, h), l=9,
                              rt=1, colorR=(255, 0, 255))

            # Draw ID above the bounding box
            cvzone.putTextRect(
                img, f'{id}', (x1, y1 - 5), scale=1, thickness=1, colorR=(0, 0, 255))

        return img

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

    def get_detections_objects(self, det, frame, frame_path):
        results = self.get_full_pred(det)
        objects = list(map(Detection, results))
        crops, img = self.soccernet.load_frame_crops(det, frame_path)
        self.siamese_network.eval()
        self.siamese_network.network.eval()
        with torch.no_grad():
            features = self.siamese_network.network.global_crop_feature(
                crops, img)

        if (len(features.shape) == 1):
            features = features.unsqueeze(dim=0)
        for i in range(len(features)):
            detection = objects[i]
            feat = features[i]
            detection.set_feature(feat)
        return objects

    def get_detections_object_kp(self, det, frame, frame_path):
        results = self.get_full_pred(det)
        objects = list(map(Detection, results))
        crops, kps = self.soccernet.load_frame_crops_kp(det, frame_path)
        self.siamese_network.eval()
        self.siamese_network.network.eval()
        with torch.no_grad():
            features = self.siamese_network.network.crop_kp(
                crops, kps
            )
        if (len(features.shape) == 1):
            features = features.unsqueeze(dim=0)
        for i in range(len(features)):
            detection = objects[i]
            feat = features[i]
            detection.set_feature(feat)
        return objects

    def process_frames_from_folderg(self, frames_folder,
                                    write_path="./logs/outputLive/",
                                    labels_write_path=None,
                                    global_text_file_path=None,
                                    write_directly=True,
                                    max_frames=None
                                    ):
        frame_files = sorted(os.listdir(frames_folder))
        total_frames = len(frame_files)
        tracker = Deep_EIoU(self.args, frame_rate=25)
        text_file = []
        with tqdm(total=total_frames, desc="Processing frames", unit="frame") as pbar:
            for frame_idx, frame_file in enumerate(frame_files):
                frame_path = os.path.join(frames_folder, frame_file)
                img = cv2.imread(frame_path)
                if (max_frames is not None and frame_idx >= max_frames):
                    break
                det = self.predict(img)
                detections = self.get_detections_object_kp(
                    det, img, frame_path)
                boxes = np.array([detection.bb for detection in detections])
                feats = np.array([detection.get_feature()
                                 for detection in detections])

                online_targets = tracker.update(boxes, feats)
                online_tlwhs = []
                online_ids = []
                online_scores = []
                results = []
                frame_name = f"frame_{frame_idx:06d}"
                frame_labels = []
                for t in online_targets:
                    tlwh = t.last_tlwh
                    tid = t.track_id
                    if tlwh[2] * tlwh[3] > self.args.min_box_area:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        result = np.concatenate((tlwh, [t.score], [0], [tid]))
                        results.append(result)
                        text_file.append(
                            f"{frame_idx},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )

                frame_labels.append(results)
                frames = self.plot_boxes(online_tlwhs, online_ids, img=img)
                if labels_write_path and results:
                    fmt = ['%.4f'] * (results[0].shape[0] - 1) + ['%d']
                    np.savetxt(f"{labels_write_path}/{frame_name}.txt",
                               results, delimiter=' ', fmt=fmt)

                if write_directly:
                    cv2.imwrite(f"{write_path}/{frame_name}.jpg", frames)

                pbar.update(1)

            with open(f"{global_text_file_path}/global.txt", 'w') as f:
                f.writelines(text_file)
