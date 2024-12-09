import numpy as np
from deep_sort import linear_assignment

from deep_sort.iou_matching import iou_cost
from siamese_tracking.track import Track
from deep_sort import iou_matching


nn_margin = 0.2


class Tracker:

    def __init__(self, metric, max_iou_distance=0.7):
        self.tracks = []
        self._next_id = 1
        self.metric = metric
        self.max_iou_distance = max_iou_distance

    def update(self, detections):
        # Get cost matrix between detections and all tracks
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)
        if (len(matches) > 0):
            self.adjust_ids(detections, matches)

        for det_idx in unmatched_detections:
            self._initiate_track(detections[det_idx])

            # Update distance metric.
        active_targets = [t.track_id for t in self.tracks]
        features, targets = [], []
        for track in self.tracks:
            features += track.features
            targets += [track.track_id for _ in track.features]
        self.metric.partial_fit(features, targets, active_targets)

    def _match(self, detections):
        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_neural(
                cost_matrix, nn_margin)
            return cost_matrix

        track_indices = [
            i for i, t in enumerate(self.tracks)]
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade_no_max_age(
                gated_metric, self.metric.matching_threshold,
                self.tracks, detections, track_indices=track_indices)

        iou_track_candidates = unmatched_tracks_a
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a
        ]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)
        matches = matches_a + matches_b
        matched_track_ids = []
        for element in matches_a:
            track_idx = element[0]
            matched_track_ids.append(self.tracks[track_idx].track_id)
        print("Appeance Matched IDs: ", len(
            matched_track_ids), [matched_track_ids])
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        track = Track(
            self._next_id,
            detection=detection
        )

        self.tracks.append(track)
        self._next_id += 1

    def adjust_ids(self, detections, matches):
        for match in matches:
            track_idx, det_idx = match
            track = self.tracks[track_idx]
            detection = detections[det_idx]
            track.update(detection)
