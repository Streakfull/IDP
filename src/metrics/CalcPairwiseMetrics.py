from deep_sort.Detection import Detection
import os
from tqdm.notebook import tqdm
import numpy as np
import torch
from deep_sort import linear_assignment
from deep_sort import iou_matching
import json


class CalcPairwiseMetrics():
    def __init__(self, gt_path, pairwise_folder, max_frames=None, max_iou_distance=0.7, conf_threshold=0.25, output_path=None) -> None:
        self.gt_path = gt_path
        # self.predict_path = predict_path
        self.labels_gt = f"{self.gt_path}/labels"
        self.max_iou_distance = max_iou_distance
        self.conf_threshold = conf_threshold
        self.outputlabels = pairwise_folder
        self.output_path = output_path

    def calc_single_frame(self, f, folder_path):
        gt, pred = self.preprocess(f, folder_path)
        matches, unmatched_gt, unmatched_pred = self.match_iou(gt, pred)
        fp = len(unmatched_pred)
        fn = len(unmatched_gt)
        tp = len(matches)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0  # Precision
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall
        res = {
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "prec": prec,
            "recall": recall,
        }
        return res

    def calc_all_ds(self, ds):
        total_fp = 0
        total_fn = 0
        total_tp = 0
        total_prec = 0
        total_rec = 0
        total = len(ds)
        with tqdm(total=total-1, desc=f"Calculating ds for conf {self.conf_threshold}", unit="frame") as pbar:
            for i in range(total):
                e = ds[i]
                f1, f2 = e["f1"], e["f2"]
                res = self.calc_pairwise_metrics(f1, f2)
                fp, fn, tp, prec, rec = res["fp"], res["fn"], res["tp"], res["prec"], res["recall"]
                total_fp += fp
                total_fn += fn
                total_tp += tp
                total_rec += rec
                total_prec += prec
                pbar.update(1)
        res = {
            "fp": total_fp,
            "fn": total_fn,
            "tp": total_tp,
            "macro_prec": total_prec/total,
            "macro_rec": total_rec/total,
            "micro_prec": total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0,
            "micro_recall": total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        }

        with open(f"{self.output_path}/conf-{self.conf_threshold}.json", "w") as json_file:
            json.dump(res, json_file, indent=4)  # indent for pretty-printing

        return res

    def calc_pairwise_metrics(self, f1, f2):
        folder_path = f"{self.outputlabels}/{f1}_{f2}"
        gtx1, predx1 = self.preprocess(f1, folder_path)
        gtx2, predx2 = self.preprocess(f2, folder_path, True)
        gt_pred_map = {}
        matches, unmatched_gt, unmatched_pred = self.match_iou(gtx1, predx1)
        for match in matches:
            gt_idx, pred_idx = match
            gt_pred_map[predx1[pred_idx].id] = gtx1[gt_idx].id
        matches, unmatched_gt, unmatched_pred = self.match_iou(gtx2, predx2)
        fp = len(unmatched_pred)
        fn = len(unmatched_gt)
        tp = 0
        for match in matches:
            gt_idx, pred_idx = match
            pred = predx2[pred_idx].id
            gt = gtx2[gt_idx].id
            mapped_id = gt_pred_map.get(pred, None)
            if (mapped_id == gt):
                tp += 1
            else:
                fp += 1
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0  # Precision
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall
        res = {
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "prec": prec,
            "recall": recall,
        }
        return res

    def preprocess(self, f, folder_path, apply_conf=False):
        pred = np.loadtxt(f"{folder_path}/{f}.txt")
        gt = np.loadtxt(f"{self.labels_gt}/frame_{f}.txt")
        gt_det = self.construct_detections(gt)
        pred_det = self.construct_detections(pred)
        if (apply_conf):
            pred_det = [
                d for d in pred_det if d.confidence >= self.conf_threshold]
        return gt_det, pred_det

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
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance,
                gt, pred)
        return matches, unmatched_gt, unmatched_pred
