import numpy as np
from torch.nn.functional import cosine_similarity
from simpletracking.Track import Track
import torch
from deep_sort.sk_learn_linear_assignment import linear_assignment


class Tracker:
    def __init__(self, max_iou_distance=0.7, max_age=500, n_init=3):
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.frame = 0
        self.tracks = []
        self._next_id = 1

    def update(self, dets):
        track_indices = list(range(len(self.tracks)))
        detection_indices = list(range(len(dets)))
        unmatched_detections = detection_indices
        det_features = np.array([dets[i].get_feature()
                                for i in detection_indices])
        target_features = np.array([self.tracks[i].get_detection().get_feature()
                                   for i in track_indices])

        cost_matrix = self.calc_cost_matrix(
            det_features=det_features, target_features=target_features)
        print("Cost matrix for frame ", self.frame, cost_matrix)
        matched_indices = self.match_detections_to_trackers(cost_matrix)
        matched_detections = matched_indices[:, 1]
        unmatched_detections = np.setdiff1d(
            detection_indices, matched_detections)
        for unmatched_detection in unmatched_detections:
            self._initiate_track(dets[unmatched_detection])

        self.update_tracks(dets, matched_indices)

        return self.tracks, matched_indices

    def calc_cost_matrix(self, det_features, target_features):
        cost_matrix = torch.zeros((len(target_features), len(det_features)))
        for i, target in enumerate(target_features):
            cost_matrix[i, :] = 1 - cosine_similarity(
                torch.Tensor(target), torch.Tensor(det_features))
        return cost_matrix.numpy()

    def _initiate_track(self, detection):
        self.tracks.append(Track(
            self._next_id, detection))
        self._next_id += 1

    def match_detections_to_trackers(self, cost_matrix):
        matched_indices = linear_assignment(cost_matrix)
        return matched_indices

    def set_frame(self, frame):
        self.frame = frame

    def update_tracks(self, dets, matched_indices):
        for idx in matched_indices:
            track_idx, det_idx = idx
            self.tracks[track_idx].set_detection(dets[det_idx])
