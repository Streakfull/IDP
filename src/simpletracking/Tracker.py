import numpy as np
from torch.nn.functional import cosine_similarity
from simpletracking.Track import Track
import torch


class Tracker:
    def __init__(self, max_iou_distance=0.7, max_age=500, n_init=3):
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

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
        targets = np.array([self.tracks[i].track_id for i in track_indices])
        cost_matrix = self.calc_cost_matrix(
            det_features=det_features, target_features=target_features)
        print(cost_matrix.shape, cost_matrix)
        for unmatched_detection in unmatched_detections:
            self._initiate_track(dets[unmatched_detection])

        return self.tracks

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
