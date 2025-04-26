from deep_sort.Detection import Detection
import os
from tqdm.notebook import tqdm
import numpy as np
import torch
from deep_sort import linear_assignment
from deep_sort import iou_matching


class CalcMetrics():
    def __init__(self,
                 gt_path,
                 predict_path,
                 max_frames=None,
                 max_iou_distance=0.7,
                 conf_threshold=None
                 ) -> None:
        self.gt_path = gt_path
        self.predict_path = predict_path
        self.labels_gt = f"{self.gt_path}/labels"
        self.labels_pred = f"{self.predict_path}/labels"
        self.max_frames = max_frames
        self.max_iou_distance = max_iou_distance
        self.map_gt_pr = {}
        self.map_pr_gt = {}
        self.map_gt_pr_all = {}
        self.temporal_map_gt_pr = {}
        self.recallf = []
        self.precisionf = []
        self.frame = 0
        self.fp_count = 0
        self.fn_count = 0
        self.gt_count = 0
        self.tp_count = 0
        self.distances = 0
        self.checked_id = 61
        self.matching_id = 20000
        self.distance = 0
        self.conf_threshold = conf_threshold

    def run(self):
        print("Calculating")

        all_labels = os.listdir(self.labels_gt)
        all_labels.sort()
        total_frames = int(all_labels[-1].split("_")[1].split(".")[0])
        if (self.max_frames is not None):
            total_frames = self.max_frames
        with tqdm(total=total_frames, desc="Calculating metrics", unit="frame") as pbar:
            for i in range(total_frames):
                gt, pred = self.pre_process(i)
                matches, unmatched_gt, unmatched_pred = self.match_iou(
                    gt, pred)
                self.fp_count += len(unmatched_pred)
                self.fn_count += len(unmatched_gt)
                self.gt_count += len(gt)
                self.add_to_maps(matches, gt, pred)
                frame_results = self.calc(matches, gt, pred)
                self.save_frame_results(frame_results)
                self.add_to_temporal_map_gt_pr(matches, gt, pred)
                self.frame += 1
                pbar.update(1)

    def pre_process(self, f_idx):
        gt = np.loadtxt(f"{self.labels_gt}/frame_{f_idx}.txt")
        pred = np.loadtxt(f"{self.labels_pred}/frame_{f_idx}.txt")
        gt_det = self.construct_detections(gt)
        pred_det = self.construct_detections(pred, is_pred=True)
        return gt_det, pred_det

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
            if is_pred and self.conf_threshold is not None:
                detections = [det for det in detections if det.get_conf() >
                              self.conf_threshold]
            return detections
        for row in outputs:
            bb = row[:-1]
            id = int(row[-1])
            detection = Detection(torch.Tensor(bb))
            detection.time_since_update = 1
            detection.id = id
            detections.append(detection)
        self.conf_threshold

        if is_pred and self.conf_threshold is not None:
            detections = [det for det in detections if det.get_conf() >
                          self.conf_threshold]
        return detections

    def match_iou(self, gt, pred):
        matches, unmatched_gt, unmatched_pred = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance,
                gt, pred)
        # gt_ids = []
        # pred_ids = []

        # for k in gt:
        #     gt_ids.append(k.id)
        # for l in pred:
        #     pred_ids.append(l.id)
        # # if (self.frame == 229):
        # #     import pdb
        # #     pdb.set_trace()
        # if (self.checked_id in gt_ids and self.checked_id in pred_ids):
        #     gt_idx_1 = gt_ids.index(self.checked_id)
        #     pred_idx_1 = pred_ids.index(self.checked_id)
        #     for match in matches:
        #         gt_p, pred_p = match
        #         if (gt_p != gt_idx_1):
        #             continue
        #         if (pred_idx_1 != pred_p):
        #             print(self.frame, "----->", pred_ids[pred_p])

        return matches, unmatched_gt, unmatched_pred

    def add_to_maps(self, matches, gt, pred):
        self.add_to_matching_map_gt(matches, gt, pred)
        self.add_to_matching_map_pred(matches, gt, pred)
        self.add_to_gt_pr_map_all(matches, gt, pred)

    def add_to_matching_map_gt(self, matches, gt, pred):
        for match in matches:
            gt_id = gt[match[0]].id
            pred_id = pred[match[1]].id
            if (gt_id in self.map_gt_pr):
                self.map_gt_pr[gt_id].append(pred_id)
                self.map_gt_pr[gt_id] = np.unique(
                    self.map_gt_pr[gt_id]).tolist()
            else:
                self.map_gt_pr[gt_id] = [pred_id]

    def add_to_matching_map_pred(self, matches, gt, pred):
        for match in matches:
            gt_id = gt[match[0]].id
            pred_id = pred[match[1]].id
            if (pred_id in self.map_pr_gt):
                self.map_pr_gt[pred_id].append(gt_id)
                self.map_pr_gt[pred_id] = np.unique(
                    self.map_pr_gt[pred_id]).tolist()
            else:
                self.map_pr_gt[pred_id] = [gt_id]

    def calc(self, matches, gt, pred):
        correct_matches = 0
        distance = 0
        for match in matches:
            gt_det, pred_det = gt[match[0]], pred[match[1]]
            is_correct = self.is_correct_match(gt_det, pred_det)
            if (is_correct):
                correct_matches += 1
            center_a = np.array([gt_det.to_xyah()[0], gt_det.to_xyah()[1]])
            center_b = np.array([pred_det.to_xyah()[0], pred_det.to_xyah()[1]])
            distance += np.linalg.norm(center_a - center_b)
        precision = self.precision(correct_matches, gt, pred)
        recall = self.recall(correct_matches, gt, pred)
        self.tp_count += correct_matches
        self.distance = + distance
        return precision, recall

    def precision(self, correct_matches, gt, pred):
        total = len(pred)
        if (total == 0):
            total = np.Infinity
        return correct_matches/total

    def recall(self, correct_matches, gt, pred):
        return correct_matches/len(gt)

    def add_to_temporal_map_gt_pr(self, matches, gt, pred):
        for match in matches:
            gt_id = gt[match[0]].id
            pred_id = pred[match[1]].id

            prev_matched_gt = self.map_pr_gt.get(pred_id, [])
            for prev_gt in prev_matched_gt:
                if (prev_gt != gt_id):
                    prev_matches = self.temporal_map_gt_pr[prev_gt]
                    next_matches = [x for x in prev_matches if x != pred_id]
                    self.temporal_map_gt_pr[prev_gt] = next_matches

            if (gt_id in self.temporal_map_gt_pr):
                self.temporal_map_gt_pr[gt_id].append(pred_id)
                self.temporal_map_gt_pr[gt_id] = np.unique(
                    self.temporal_map_gt_pr[gt_id]).tolist()
            else:
                self.temporal_map_gt_pr[gt_id] = [pred_id]

    def add_to_gt_pr_map_all(self, matches, gt, pred):
        for match in matches:
            gt_id = gt[match[0]].id
            pred_id = pred[match[1]].id
            # if (gt_id == self.checked_id and pred_id != self.matching_id):
            #    # print(match)
            #     # import pdb
            #     # pdb.set_trace()
            #     print(self.frame, match,  "----->", pred_id)
            if (gt_id in self.map_gt_pr_all):
                self.map_gt_pr_all[gt_id].append(pred_id)
            else:
                self.map_gt_pr_all[gt_id] = [pred_id]

    def save_frame_results(self, frame_results):
        precision, recall = frame_results
        self.precisionf.append(precision)
        self.recallf.append(recall)

    def is_correct_match(self, gt, pred):
        gt_id, pred_id = gt.id, pred.id
        all_gt_matches = self.map_pr_gt.get(pred_id, [])
        if (len(all_gt_matches) <= 1):
            return True
        current_temporal_match = self.temporal_map_gt_pr.get(gt_id, [])
        if (pred_id in current_temporal_match):
            return True

        return False

    def get_results(self):
        precision = np.array(self.precisionf)
        recall = np.array(self.recallf)
        results = {
            "precision": np.average(precision),
            "recall": np.average(recall),
            "GT": len(list(self.map_gt_pr.keys())),
            "idsw": self.count_total_id_switchs(),
            "FP": self.fp_count,
            "FN": self.fn_count,
            "MOTP": self.distance/(self.tp_count+np.finfo(float).eps),
            "fragments": self.count_total_id_fragments(),
            "id_switches_per_gt":  self.get_id_switches_map(),
            # "gt_pr": self.map_gt_pr_all

        }
        results["MOTA"] = 1 - (results["FP"]+results["FN"] +
                               results["idsw"])/(self.gt_count+np.finfo(float).eps)
        return results
        # results["mot"]

    def count_total_id_switchs(self):
        id_switches = 0
        gt = list(self.map_gt_pr_all.keys())
        for key in gt:
            pred = self.map_gt_pr_all[key]
            id_switches += self.count_id_switches(pred)
        return id_switches

    def count_id_switches(self, ids):
        id_switches = 0
        for i in range(1, len(ids)):
            if ids[i] != ids[i - 1]:  # Check if the current ID is different from the previous one
                id_switches += 1
        return id_switches

    def count_total_id_fragments(self):
        id_fragments = 0
        for gt_id, pred_ids in self.map_gt_pr_all.items():
            id_fragments += self.count_id_fragments(pred_ids)
        return id_fragments

    def count_id_fragments(self, pred_ids):
        fragments = 0
        previous_id = None
        for pred_id in pred_ids:
            if pred_id != previous_id:  # Fragment detected
                fragments += 1
                previous_id = pred_id
        return fragments

    def get_id_switches_map(self):
        id_switches_map = {}
        gt = list(self.map_gt_pr_all.keys())  # List of ground truth IDs
        for key in gt:
            pred = self.map_gt_pr_all[key]
            id_switches_map[key] = self.count_id_switches(
                pred)  # Count ID switches for each GT ID
        return id_switches_map
