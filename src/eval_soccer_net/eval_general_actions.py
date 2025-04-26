import json
import os
import copy
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


class EvalSoccernetPredictions():
    def __init__(self, pred_file, gt_path, out_folder, nms_window=2):
        self.pred_file = pred_file
        self.out_folder = out_folder
        self.nms_window = nms_window
        self.gt = gt_path

    def _load_json(self, file_path):
        """Load JSON file from the given path."""
        with open(file_path, "r") as f:
            return json.load(f)

    def eval(self):
        preds = self._load_json(pred_file)
        gts = self._load_json(self.gt)
        # Everything should be in a seperate function
        # First calculate precision/recall/accuracy/f1score with 5 frame tolerance
        # And no conf threshold
        # Then calculate the precision for 0,1,2,3,4,5 tolerance
        # Then plot this as a curve
        # Then go from conf threshhold 0-->1
        # give the PR curve based on this conf threshold
        # the pr curve should start with 0 recall and 1 precision
        # and end with 0 precision and 1 recall
        # No need to perfom non-maximum supression, I already this and you can assume that
        # the pred file is alread-preprocessed

        video_map = {g['video']: g for g in gts}

        print("Calculating metrics at 5-frame tolerance (no conf threshold)...")
        precision, recall, accuracy, f1 = self.evaluate_frame_tolerance(
            preds, video_map, tolerance=5)
        print(
            f"Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

        print("Calculating PR curve for tolerances 0-5...")
        self.plot_pr_curve_by_tolerance(preds, video_map)

        print("Calculating PR curve for confidence thresholds 0.0 to 1.0...")
        self.plot_pr_curve_by_confidence(preds, video_map, tolerance=5)

    def match_events(self, pred_events, gt_events, tolerance):
        matched_gt = set()
        tp = 0
        for pe in pred_events:
            for i, ge in enumerate(gt_events):
                if i in matched_gt:
                    continue
                if pe['label'] == ge['label'] and abs(pe['frame'] - ge['frame']) <= tolerance:
                    # print(pe['label'], "LABEL")
                    tp += 1
                    matched_gt.add(i)
                    break
        fp = len(pred_events) - tp
        fn = len(gt_events) - tp
        return tp, fp, fn

    def evaluate_frame_tolerance(self, preds, gt_map, tolerance=5):
        total_tp, total_fp, total_fn = 0, 0, 0
        for video_pred in preds:
            video = video_pred['video']
            if video not in gt_map:
                continue
            pred_events = video_pred['events']
            gt_events = gt_map[video]['events']
            tp, fp, fn = self.match_events(pred_events, gt_events, tolerance)
            total_tp += tp
            total_fp += fp
            total_fn += fn
        precision = total_tp / (total_tp + total_fp + 1e-8)
        recall = total_tp / (total_tp + total_fn + 1e-8)
        accuracy = total_tp / (total_tp + total_fp + total_fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return precision, recall, accuracy, f1

    def plot_pr_curve_by_tolerance(self, preds, gt_map):
        tolerances = list(range(6))
        precisions, recalls = [], []

        for t in tolerances:
            precision, recall, _, _ = self.evaluate_frame_tolerance(
                preds, gt_map, tolerance=t)
            precisions.append(precision)
            recalls.append(recall)

        # Precision vs Tolerance
        plt.figure()
        plt.plot(tolerances, precisions, marker='o', label='Precision')
        plt.title("Precision vs Frame Tolerance")
        plt.xlabel("Tolerance (frames)")
        plt.ylabel("Precision")
        plt.xticks(tolerances)
        plt.grid(True)
        plt.savefig(os.path.join(self.out_folder,
                    "precision_vs_tolerance.png"))
        plt.close()

        # Recall vs Tolerance
        plt.figure()
        plt.plot(tolerances, recalls, marker='o',
                 label='Recall', color='green')
        plt.title("Recall vs Frame Tolerance")
        plt.xlabel("Tolerance (frames)")
        plt.ylabel("Recall")
        plt.xticks(tolerances)
        plt.grid(True)
        plt.savefig(os.path.join(self.out_folder, "recall_vs_tolerance.png"))
        plt.close()

    def plot_pr_curve_by_confidence(self, preds, gt_map, tolerance=5):
        thresholds = np.linspace(0.0, 1.0, 50)
        precisions, recalls = [], []

        # Calculate precision and recall for each confidence threshold
        for thresh in thresholds:
            filtered_preds = copy.deepcopy(preds)
            for v in filtered_preds:
                v['events'] = [e for e in v['events'] if e['score'] >= thresh]
                v['num_events'] = len(v['events'])
            precision, recall, _, _ = self.evaluate_frame_tolerance(
                filtered_preds, gt_map, tolerance)
            precisions.append(precision)
            recalls.append(recall)

        # Enforce PR curve bounds
        # recalls = [0.0] + recalls + [1.0]
        # precisions = [1.0] + precisions + [0.0]
        # thresholds = [0.0] + thresholds.tolist() + [1.0]

        precisions[-1] = 1.0

        # Plotting the PR curve
        plt.figure()
        plt.plot(recalls, precisions, marker='.', label="PR Curve")

        # Annotate threshold values on the plot
        for i in range(0, len(thresholds), len(thresholds)//10):  # Annotate 10 points
            plt.annotate(f'{thresholds[i]:.2f}', (recalls[i], precisions[i]),
                         textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8)

        plt.title("Precision-Recall Curve by Confidence Threshold")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.grid(True)
        plt.savefig(os.path.join(self.out_folder,
                    "pr_curve_confidence_with_labels.png"))
        plt.close()

    def get_nms_pred(self):
        """Load predictions, apply NMS, and save results."""
        predictions = self._load_json(self.pred_file)
        filtered_predictions = self.non_maximum_suppression(
            predictions, self.nms_window)

        output_file = os.path.join(self.out_folder, "nms_pred.json")
        with open(output_file, "w") as f:
            json.dump(filtered_predictions, f, indent=4)

        print(f"Filtered predictions saved to {output_file}")
        return filtered_predictions

    def non_maximum_suppression(self, pred, window):
        """Perform Non-Maximum Suppression (NMS) on event predictions."""
        new_pred = []
        for video_pred in pred:
            events_by_label = defaultdict(list)
            for e in video_pred['events']:
                events_by_label[e['label']].append(e)

            events = []
            for v in events_by_label.values():
                for e1 in v:
                    for e2 in v:
                        if (
                            e1['frame'] != e2['frame']
                            and abs(e1['frame'] - e2['frame']) <= window
                            and e1['score'] < e2['score']
                        ):
                            break  # Higher score event found
                    else:
                        events.append(e1)

            events.sort(key=lambda x: x['frame'])
            new_video_pred = copy.deepcopy(video_pred)
            new_video_pred['events'] = events
            new_video_pred['num_events'] = len(events)
            new_pred.append(new_video_pred)
        return new_pred


# pred_file = "./eval_soccer_net/nms_pred-ga.json"
# gt_file = "./eval_soccer_net/test.json"
# out_folder = "./eval_soccer_net"

# e = EvalSoccernetPredictions(pred_file, gt_file, out_folder)
# # e.get_nms_pred()
# e.eval()


pred_file = "./eval_soccer_net/nms_predba.json"
gt_file = "./eval_soccer_net/testba.json"
out_folder = "./eval_soccer_net"

e = EvalSoccernetPredictions(pred_file, gt_file, out_folder)
# e.get_nms_pred()
e.eval()
