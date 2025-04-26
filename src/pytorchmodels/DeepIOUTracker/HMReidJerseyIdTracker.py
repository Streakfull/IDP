from termcolor import cprint
import re
import os
from tabnanny import verbose
import cvzone
from hm_sort import Strack
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
from hm_sort_jid.tracker import BaseTrack, Deep_EIoU
from hm_sort_jid.reid.torchreid.utils import FeatureExtractor
from siamese_tracking.Detection import Detection
from pytorchmodels.SiameseTriplet import TripletSiamese
from pytorchmodels.jerseyId.jerseyId import JerseyID
from datasets.JerseyId import JerseyDataset


class HMTracker(ObjectDetection):
    def __init__(self, write_path, use_high_res=False, res=None) -> None:
        self.use_high_res = use_high_res
        super().__init__("")
        self.args = make_parser().parse_args()
        self.res = res
        with open(self.configs_path, "r") as in_file:
            self.global_configs = yaml.safe_load(in_file)
            self.siamese_configs = self.global_configs["model"]["siamese"]
            self.siamese_network = self.load_siamese()
            self.jersey_id = self.load_jersey_id()
        self.soccernet = SoccerNetDataset(
            "./raw_dataset/soccernet-tracking/raw/tracking/train2.txt", train=False)
        # self.extractor = FeatureExtractor(
        #     model_name='osnet_x1_0',
        #     model_path='chkpts2/deep_Iou/sports_model.pth.tar-60',
        #     device='cuda'
        # )

    def load_jersey_id(self):
        ckpt_path = "../logs/training/jersey-id/trainTest/lc-resnet34-multiclass/2025_03_20_06_02_25/checkpoints/epoch-latest.ckpt"
        jersey_configs = self.global_configs["model"]["jerseyId"]
        model = JerseyID(jersey_configs)
        model.load_ckpt(ckpt_path)
        model.to("cuda:0")
        return model

    def load_siamese(self):
        self.siamese_network = Siamese(
            self.siamese_configs, self.global_configs["training"])
        # self.siamese_network = TripletSiamese(self.siamese_configs)
        self.siamese_network.load_ckpt(
            self.global_configs["training"]["ckpt_path"])
        device = "cuda:0"
        self.siamese_network.to(device)
        return self.siamese_network

    def load_model(self):
        if (self.use_high_res):
            return self.load_model_high_res()
        weights = "./runs/detect/fintune-soccernet/weights/best.pt"
        model = YOLO("yolo11x.pt", verbose=False)
        self.configs_path = "./configs/global_configs.yaml"
        model.fuse()
        self.model = model
        self.model.model._predict_once = MethodType(_predict_once, model.model)
        _ = self.model(ASSETS / "bus.jpg", save=False, embed=[16, 19, 22, 23])
        return model

    def load_model_high_res(self):
        weights = "./runs/detect/fintune-soccernet/weights/best.pt"
        model = YOLO(weights, verbose=False)
        self.configs_path = "./configs/global_configs.yaml"
        model.fuse()
        self.model = model
        return model

    def predict(self, img):
        if (self.use_high_res):
            imgsz = self.res if self.res is not None else 1920
            result = self.model.predict(
                img, imgsz=imgsz, verbose=False, classes=[0])
            return result[0]
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
        # boxes = self.filter_ball(boxes)
        detections = torch.cat((boxes, conf, cls), dim=1).cpu()
        return detections

    # def get_full_pred(self, det):
    #     try:
    #         boxes = det.boxes.xyxy
    #     except:
    #         import pdb
    #         pdb.set_trace()

    #     cls = det.boxes.cls.unsqueeze(1)  # Shape: [N, 1]
    #     conf = det.boxes.conf.unsqueeze(1)  # Shape: [N, 1]

    #     # Filter boxes and get valid indices
    #     boxes, valid_indices = self.filter_ball(boxes)

    #     if boxes.shape[0] == 0:  # If no boxes remain after filtering
    #         return torch.tensor([]), torch.tensor([], dtype=torch.bool)

    #     # Select the corresponding cls and conf values based on valid indices
    #     cls = cls[valid_indices]
    #     conf = conf[valid_indices]

    #     # Stack the final predictions
    #     detections = torch.cat((boxes, conf, cls), dim=1).cpu()

        return detections, valid_indices

    def get_detections_objects(self, det, frame, frame_path):
        results = self.get_full_pred(det)
        objects = list(map(Detection, results))
        crops, img = self.soccernet.load_frame_crops(det, frame_path)
        self.siamese_network.eval()
        self.siamese_network.clip.eval()
        if len(crops) == 0:
            # You can decide whether to return an empty list or some other behavior
            return objects  # or return an empty list if needed: return []
        with torch.no_grad():
           # features = self.extractor(crops)
            # import pdb
            # pdb.set_trace()
            features = self.siamese_network.img_feature(crops)

        if (len(features.shape) == 1):
            features = features.unsqueeze(dim=0)
        for i in range(len(features)):
            detection = objects[i]
            feat = features[i]
            detection.set_feature(feat)
        return objects

    def get_detections_objects_file(self, det, frame, frame_path):
        crops = self.soccernet.load_frame_crops_det(det, frame_path)
        self.siamese_network.eval()
        self.siamese_network.clip.eval()
        if len(crops) == 0:
            return det  # or return an empty list if needed: return []
        with torch.no_grad():
            features = self.siamese_network.img_feature(crops)

        if (len(features.shape) == 1):
            features = features.unsqueeze(dim=0)
        for i in range(len(features)):
            feat = features[i]
            det[i].set_feature(feat)
        return det

    def filter_ball(self, det):
        HEIGHT_MIN = 30
        WIDTH_MIN = 25
        filtered_detections = []
        valid_indices = []

        for idx, detection in enumerate(det):
            # Assuming detections are in [x1, y1, x2, y2] format
            x1, y1, x2, y2 = detection
            width = x2 - x1
            height = y2 - y1

            if width >= WIDTH_MIN and height >= HEIGHT_MIN:
                filtered_detections.append(detection)
                # Store the index of valid detections
                valid_indices.append(idx)
            else:
                cprint("Ball Detected", color="yellow")

        return torch.stack(filtered_detections) if filtered_detections else torch.tensor([]), valid_indices

        # Filter detections here  if detection dimensions less than min width and height and return the final list
        return filtered_detections

    def get_detections_object_kp(self, det, frame, frame_path):
        results, valid_indices = self.get_full_pred(det)
        objects = list(map(Detection, results))
        if (len(valid_indices) == 0):
            return []
        crops, kps = self.soccernet.load_frame_crops_kp(
            det[valid_indices], frame_path)
        if len(crops) == 0:
            # You can decide whether to return an empty list or some other behavior
            return objects  # or return an empty list if needed: return []
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
                                    max_frames=None,
                                    sequence_name=None
                                    ):
        frame_files = sorted(os.listdir(frames_folder))
        total_frames = len(frame_files)
        tracker = Deep_EIoU(self.args, frame_rate=25)
        text_file = []
        with tqdm(total=total_frames, desc="Processing frames", unit="frame") as pbar:
            for frame_idx, frame_file in enumerate(frame_files):
                # if (frame_idx < 150):
                #     tracker.frame_id += 1
                #     continue
                frame_path = os.path.join(frames_folder, frame_file)
                img = cv2.imread(frame_path)
                if (max_frames is not None and frame_idx >= max_frames):
                    break
                det = self.predict(img)

                detections = self.get_detections_objects(
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
                            f"{frame_idx+1},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )

                frame_labels.append(results)
                frames = self.plot_boxes(online_tlwhs, online_ids, img=img)
                # if labels_write_path and results:
                #     fmt = ['%.4f'] * (results[0].shape[0] - 1) + ['%d']
                #     np.savetxt(f"{labels_write_path}/{frame_name}.txt",
                #                results, delimiter=' ', fmt=fmt)

                if write_directly:
                    cv2.imwrite(f"{write_path}/{frame_name}.jpg", frames)

                pbar.update(1)
            print("Total IDs:", BaseTrack._count)
            with open(f"{global_text_file_path}/{sequence_name}.txt", 'w') as f:
                f.writelines(text_file)

    def process_frames_from_sorted(self, frames_folder,
                                   write_path="./logs/outputLive/",
                                   labels_write_path=None,
                                   global_text_file_path=None,
                                   write_directly=True,
                                   max_frames=None,
                                   sequence_name=None
                                   ):
        frame_files = sorted(os.listdir(frames_folder))

        def extract_frame_number(filename):
            match = re.search(r'frame_(\d+)\.jpg', filename)
            return int(match.group(1)) if match else float('inf')

        # Sorting based on extracted frame number
        frame_files = sorted(frame_files, key=extract_frame_number)
        total_frames = len(frame_files)
        tracker = Deep_EIoU(self.args, frame_rate=25)
        text_file = []
        with tqdm(total=total_frames, desc="Processing frames", unit="frame") as pbar:
            for frame_idx, frame_file in enumerate(frame_files):
                # if (frame_idx < 150):
                #     tracker.frame_id += 1
                #     continue
                frame_path = os.path.join(frames_folder, frame_file)
                img = cv2.imread(frame_path)
                if (max_frames is not None and frame_idx >= max_frames):
                    break
                det = self.predict(img)
                # det = self.filter_ball(det)
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
                            f"{frame_idx+1},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )

                frame_labels.append(results)
                frames = self.plot_boxes(online_tlwhs, online_ids, img=img)
                # if labels_write_path and results:
                #     fmt = ['%.4f'] * (results[0].shape[0] - 1) + ['%d']
                #     np.savetxt(f"{labels_write_path}/{frame_name}.txt",
                #                results, delimiter=' ', fmt=fmt)

                if write_directly:
                    cv2.imwrite(f"{write_path}/{frame_name}.jpg", frames)

                pbar.update(1)
            print("Total IDs:", BaseTrack._count)
            with open(f"{global_text_file_path}/{sequence_name}.txt", 'w') as f:
                f.writelines(text_file)

    def get_jersey_ids(self, det, frame, frame_path):
        crops = JerseyDataset.load_frame_crops_det(
            det, frame_path=frame_path)
        self.jersey_id.eval()
        if len(crops) == 0:
            return
        jids, mask = self.jersey_id.inference_unlabeled(crops)
        jidcounter = 0
        for i, m in enumerate(mask):
            if (m):
                det[i].set_jid(jids[jidcounter].item())
                jidcounter += 1

    def process_frames_from_folder_fixed_bb(self, frames_folder,
                                            write_path="./logs/outputLive/",
                                            labels_write_path=None,
                                            global_text_file_path=None,
                                            write_directly=True,
                                            max_frames=None,
                                            sequence_name=None,
                                            bb_folder=None
                                            ):
        frame_files = sorted(os.listdir(frames_folder))
        total_frames = len(frame_files)
        tracker = Deep_EIoU(self.args, frame_rate=25)
        text_file = []
        bb_file = f"{bb_folder}/{sequence_name}.txt"
        with tqdm(total=total_frames, desc="Processing frames", unit="frame") as pbar:
            for frame_idx, frame_file in enumerate(frame_files):
                # if (frame_idx < 150):
                #     tracker.frame_id += 1
                #     continue
                frame_path = os.path.join(frames_folder, frame_file)
                img = cv2.imread(frame_path)
                if (max_frames is not None and frame_idx >= max_frames):
                    break

                # det = self.predict(img)  # Old we need to replace this
                det = self.read_detections_from_file(bb_file, frame_idx+1)
                detections = self.get_detections_objects_file(
                    det, img, frame_path)
                self.get_jersey_ids(detections, img, frame_path)
                boxes = np.array([detection.bb for detection in detections])
                feats = np.array([detection.get_feature()
                                 for detection in detections])
                jids = np.array([detection.jid for detection in detections])

                online_targets = tracker.update(boxes, feats, jids)
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
                            f"{frame_idx+1},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )

                frame_labels.append(results)
                frames = self.plot_boxes(online_tlwhs, online_ids, img=img)
                # if labels_write_path and results:
                #     fmt = ['%.4f'] * (results[0].shape[0] - 1) + ['%d']
                #     np.savetxt(f"{labels_write_path}/{frame_name}.txt",
                #                results, delimiter=' ', fmt=fmt)

                if write_directly:
                    cv2.imwrite(f"{write_path}/{frame_name}.jpg", frames)

                pbar.update(1)
            print("Total IDs:", BaseTrack._count)
            with open(f"{global_text_file_path}/{sequence_name}.txt", 'w') as f:
                f.writelines(text_file)

    def read_detections_from_file(self, bb_file, frame_idx):
        """Reads bounding box detections from a file for a given frame."""
        detections = []
        with open(bb_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            values = line.strip().split(',')
            if int(values[0]) == frame_idx:
                x, y, w, h = map(float, values[2:6])
                score = float(values[6])
                class_id = 0  # Default class since the file lacks class info

                # Convert (x, y, w, h) -> (x1, y1, x2, y2)
                x1, y1, x2, y2 = x, y, x + w, y + h

                # Create a detection in expected format: (x1, y1, x2, y2, conf, cls)
                detection = Detection(
                    np.array([x1, y1, x2, y2, score, class_id]))
                detections.append(detection)

        return detections
