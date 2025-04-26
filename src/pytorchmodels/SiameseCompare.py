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
import cv2
import cvzone
from PIL import Image
from deep_sort.sk_learn_linear_assignment import linear_assignment
from pytorchmodels.Siamese import Siamese
from training import ModelBuilder
import yaml
from deep_sort import iou_matching
from deep_sort.linear_assignment import min_cost_matching
import os

# visual_siamese = "../logs/training/siamese-resnet/visual_samples_100k/2024_11_14_03_02_33/checkpoints/epoch-best.ckpt"
visual_siamese = "../logs/training/week7/train_full/train_full/2024_11_28_14_40_42/checkpoints/epoch-best.ckpt"
visual_siamese = "../logs/training/week8/20kFarDataset/train_full/2024_12_01_04_12_28/checkpoints/epoch-best.ckpt"

transform = transforms.Compose([
    # Resize to 224x224 (or your target size)
    transforms.Resize((224, 224)),
    transforms.ToTensor(),          # Convert image to tensor
    # Normalize for pre-trained models
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
        0.229, 0.224, 0.225])
])


class SiamaeseCompare():

    def __init__(self,
                 gt_path="../logs/benchmarks/clip_1/corrected",
                 max_iou_distance=0.9,
                 eval_write_path=None,
                 conf_read_path=None,
                 raw_frames_path=None
                 ):
        self.yolo = self.load_model()
        self.configs_path = "./configs/global_configs.yaml"
        self.labels_gt = f"{gt_path}/labels"
        self.gt_path = gt_path
        self.max_iou_distance = max_iou_distance
        self.eval_write_path = eval_write_path
        self.conf_read_path = conf_read_path
        self.raw_frames_path = raw_frames_path
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

    def process_pair_frames(self, f1, f2, cos_distance=0.2):
        d_a, img_a = self.process_frame(f1)
        d_b, img_b = self.process_frame(f2)
        cost_matrix = self.get_cost_matrix(
            (d_a), d_b, cos_distance_threshold=cos_distance)
        matches, unmatched_a, unmatched_b = self.min_cost_matching(
            cost_matrix, d_a, d_b)
        d_a, d_b = self.adjust_ids(d_a, d_b,
                                   matches,
                                   unmatched_a,
                                   unmatched_b,
                                   cost_matrix)
        img_a, img_b = self.plot_boxes(
            (d_a), img_a), self.plot_boxes(d_b, img_b)
        confusion_matrix = self.calc_metrics(f1, f2, d_a, d_b)
        # confusion_matrix = None
        return {
            "cost_matrix": cost_matrix,
            "matches": matches,
            "u_a": unmatched_a,
            "u_b": unmatched_b,
            "img_a": img_a,
            "img_b": img_b,
            "confusion_matrix": confusion_matrix
        }

    def get_tp_pairs(self, pairs):
        # pairs = [pairs[0]]
        for idx, pair in tqdm(enumerate(pairs), total=len(pairs), desc="Processing TP pairs"):
            n_a, n_b = pair
            conf_dir = f"{self.conf_read_path}/conf_logs/{n_a}-{n_b}/tp_log.txt"
            with open(conf_dir, "r") as f:
                cleaned_data = [line.strip().replace("(", "").replace(")", "")
                                for line in f]
            tp_log = np.array([tuple(map(int, line.split(',')))
                              for line in cleaned_data])
            gta = np.loadtxt(f"{self.labels_gt}/frame_{n_a}.txt")
            gtb = np.loadtxt(f"{self.labels_gt}/frame_{n_b}.txt")
            deta = self.construct_detections(gta)
            detb = self.construct_detections(gtb)
            fa = self.read_img_frame(n_a)
            fb = self.read_img_frame(n_b)
            det_pairs = []
            for p in tp_log:
                gt_id, det_id = p
                pa = find_object_by_id(deta, gt_id)
                pb = find_object_by_id(detb, gt_id)
                if (pa is None or pb is None):
                    continue
                assert pa is not None and pb is not None
                det_pairs.append((pa, pb))

            self.save_cropped_images(
                det_pairs, fa, fb, self.eval_write_path, n_a, n_b)

    def get_fn_pairs(self, pairs):
        for idx, pair in tqdm(enumerate(pairs), total=len(pairs), desc="Processing FN pairs"):
            n_a, n_b = pair.split("-")
            n_a, n_b = int(n_a), int(n_b)
            conf_dir = f"{self.conf_read_path}/conf_logs/{n_a}-{n_b}/fn_log.txt"
            with open(conf_dir, "r") as f:
                cleaned_data = [line.strip().replace("(", "").replace(")", "")
                                for line in f]
            fn_log = np.array([tuple(map(int, line.split(',')))
                               for line in cleaned_data])

            gta = np.loadtxt(f"{self.labels_gt}/frame_{n_a}.txt")
            gtb = np.loadtxt(f"{self.labels_gt}/frame_{n_b}.txt")

            deta = self.construct_detections(gta)
            detb = self.construct_detections(gtb)
            fa = self.read_img_frame(n_a)
            fb = self.read_img_frame(n_b)
            det_pairs = []
            for p in fn_log:
                gt_id, _ = p
                pa = find_object_by_id(deta, gt_id)
                pb = find_object_by_id(detb, gt_id)
                if (pa is None or pb is None):
                    continue
                assert pa is not None and pb is not None
                det_pairs.append((pa, pb))
            self.save_cropped_images(
                det_pairs, fa, fb, self.eval_write_path, n_a, n_b)

    def get_fp_pairs(self, pairs):
        for idx, pair in tqdm(enumerate(pairs), total=len(pairs), desc="Processing FP pairs"):
            n_a, n_b = pair.split("-")
            n_a, n_b = int(n_a), int(n_b)
            conf_dir = f"{self.conf_read_path}/conf_logs/{n_a}-{n_b}/fp_log.txt"
            with open(conf_dir, "r") as f:
                cleaned_data = [line.strip().replace("(", "").replace(")", "")
                                for line in f]
            fp_log = np.array([tuple(map(int, line.split(',')))
                               for line in cleaned_data])
            gta = np.loadtxt(f"{self.labels_gt}/frame_{n_a}.txt")
            gtb = np.loadtxt(f"{self.labels_gt}/frame_{n_b}.txt")
            detb = self.construct_detections(gtb)
            fa_path = f"{self.raw_frames_path}/frame_{n_a}.jpg"
            fa = self.read_img_frame(n_a)
            fb = self.read_img_frame(n_b)
            det_pairs = []
            for p in fp_log:
                gt_id, det_id = p
                pb = find_object_by_id(detb, gt_id)
                deta, _ = self.process_frame(fa_path)
                for i, det in enumerate(deta):
                    det.id = i
                pa = find_object_by_id(deta, det_id)
                if (pa is None or pb is None):
                    continue
                assert pa is not None and pb is not None
                det_pairs.append((pa, pb))

            self.save_cropped_images(
                det_pairs, fa, fb, self.eval_write_path, n_a, n_b)

    def save_cropped_images(self, det_pairs, fa, fb, output_dir, na, nb):
        output_dir = f"{output_dir}/{na}-{nb}"
        os.makedirs(output_dir, exist_ok=True)
        if (len(det_pairs) == 0):
            return
        for i, p2 in enumerate(det_pairs):
            bba, bbb = p2
            # Get cropped images
            bba_img, bbb_img = get_img_crop_from_frame(
                bba.xyxy, fa), get_img_crop_from_frame(bbb.xyxy, fb)

            # Save images to the directory with unique filenames
            bba_filename = os.path.join(output_dir, f"bba_{i}.png")
            bbb_filename = os.path.join(output_dir, f"bbb_{i}.png")
            bba_img.save(bba_filename)
            bbb_img.save(bbb_filename)

        print(f"Saved: {bba_filename}, {bbb_filename}")

    def read_img_frame(self, f):
        img = f"{self.raw_frames_path}/frame_{f}.jpg"
        img = Image.open(img).convert("RGB")
        img = np.array(img)
        return img

    def find_object_by_id(objects, target_id):
        for obj in objects:
            if obj.get("id") == target_id:
                return obj
        return None

    def calc_metrics(self, f1, f2, d_a, d_b):
        ########################## Frame A ################################
        n_a = extract_frame_number(f1)
        n_b = extract_frame_number(file_path=f2)
        gt = np.loadtxt(f"{self.labels_gt}/frame_{n_a}.txt")
        gt_det = self.construct_detections(gt)
        matches, unmatched_gt, unmatched_pred = self.match_iou(gt_det, d_a)
        id_map = self.construct_id_map(gt_det, d_a, matches)
        # import pdb
        # pdb.set_trace()
        ########################## Frame B ################################
        gtb = np.loadtxt(f"{self.labels_gt}/frame_{n_b}.txt")
        gtb_det = self.construct_detections(gtb)
        matchesb, _, _ = self.match_iou(gtb_det, d_b)
        confusion_matrix = self.get_confusion_matrix(
            gtb_det, d_b, matchesb, id_map)
        self.write_confusion_matrix(confusion_matrix, n_a, n_b)
        return confusion_matrix

    def get_confusion_matrix(self, gt, pred, matches, id_map):
        tp, fp, fn, tn = 0, 0, 0, 0
        tp_log, fp_log, fn_log = [], [], []
        for match in matches:
            gt_idx, pred_idx = match
            p_det_id = pred[pred_idx].id
            gt_id = gt[gt_idx].id
            if p_det_id == -1:
                if gt_id in id_map.values():
                    fn += 1
                    fn_log.append((gt_id, pred_idx))
                else:
                    tn += 1
                continue
            if (p_det_id not in id_map):
                tp_log.append((gt_id, p_det_id))
                tp += 1
                continue

            pred_gt_id = id_map[p_det_id]
            if (pred_gt_id == gt_id):
                tp += 1
                tp_log.append((gt_id, p_det_id))
            else:
                fp += 1
                fp_log.append((gt_id, p_det_id))
        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "tp_log": tp_log,
            "fp_log": fp_log,
            "fn_log": fn_log
        }

    def write_confusion_matrix(self, confusion_matrix, fa, fb):
        if self.eval_write_path is None:
            return

        # Create directory in the form 'fa-fb' under eval_write_path
        directory = os.path.join(self.eval_write_path, f"{fa}-{fb}")
        os.makedirs(directory, exist_ok=True)

        # Save confusion matrix metrics (tp, fp, fn, tn) to confusion_matrix.npy
        metrics = {
            "tp": confusion_matrix["tp"],
            "fp": confusion_matrix["fp"],
            "fn": confusion_matrix["fn"],
            "tn": confusion_matrix["tn"]
        }
        np.save(os.path.join(directory, "confusion_matrix.npy"), metrics)

        # Save logs (tp_log, fp_log, fn_log) to respective files

        for log_name in ["tp_log", "fp_log", "fn_log"]:
            log_data = confusion_matrix[log_name]
            log_file_path = os.path.join(directory, f"{log_name}.txt")
            with open(log_file_path, "w") as f:
                for log_entry in log_data:
                    f.write(f"{log_entry}\n")

    def construct_id_map(self, gt, pred, matches):
        id_map = {}
        for match in matches:
            gt_idx, pred_idx = match
            id_map[pred[pred_idx].id] = gt[gt_idx].id
        return id_map

    def construct_detections(self, outputs):
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
            detection = Detection(torch.Tensor(bb))
            detection.time_since_update = 1
            detection.id = id
            detections.append(detection)
        return detections

    def match_iou(self, gt, pred):
        matches, unmatched_gt, unmatched_pred = \
            min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance,
                gt, pred)
        return matches, unmatched_gt, unmatched_pred

    def process_frame(self, img):
        img = Image.open(img).convert("RGB")
        img = np.array(img)
        img_copy = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        det = self.predict((img))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detections = self.get_detections_objects(det, img_copy)

        return detections, img

    def get_cost_matrix(self, d_a, d_b, cos_distance_threshold=0.2):
        with torch.no_grad():
            self.siamese_network.eval()
            f_a = [d.feature for d in d_a]
            f_b = [d.feature for d in d_b]
            a_ids = [d.feature for d in d_a]
            cost_matrix = torch.zeros((len(d_a), len(d_b)))
            for i, target in enumerate(f_a):
                dis = torch.nn.functional.cosine_similarity(
                    torch.Tensor(target), torch.Tensor(f_b))
                # f_bt = torch.Tensor(f_b).to("cuda:0")
                # tgt = torch.Tensor(target)
                # bs = f_bt.shape[0]
                # tgt = torch.Tensor.repeat(tgt, bs, 1).to("cuda:0")
                # fc = self.siamese_network.network.classif(tgt, f_bt)

                # fc[fc <= 0.5] = 0
                # fc[fc > 0.5] = 1
                # print((fc == 1).shape)
                # # dis[(fc <= 0.5).flatten()] = -1
                cost_matrix[i, :] = 1 - dis

            # print(cost_matrix)
            cost_matrix[cost_matrix > cos_distance_threshold] = 1e5
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

            if (conf > 0.25):
                # cvzone.putTextRect(
                #     img, f'{id}, {conf}', (x1, y1), scale=1, thickness=1, colorR=(0, 0, 255))
                cvzone.putTextRect(
                    img, f'{id}', (x1, y1-5), scale=1, thickness=1, colorR=(0, 0, 255))
                cvzone.cornerRect(img, (x1, y1, w, h), l=9,
                                  rt=1, colorR=(255, 0, 255))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def eval(self, batch, decision_threshold=0.5):
        with torch.no_grad():
            self.siamese_network.eval()
            _, _, pred = self.siamese_network(batch)
            pred = torch.nn.functional.sigmoid(pred)
            # pred[pred >= decision_threshold] = 1
            # pred[pred <= decision_threshold] = 0
            return pred
