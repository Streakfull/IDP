# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track

nn_margin = 0.2


class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=30, n_init=3, print_cost_matrix=False, use_kalman=True, start_confirmed=False):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        # self.max_age = 1000
        self.n_init = n_init
        self.print_cost_matrix = print_cost_matrix

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1
        self.use_kalman = use_kalman
        self.start_confirmed = start_confirmed

        self.metrics = {
            "removed": 0,
            "id_frames": {

            },
            "new_ids": 0
        }

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)
        # if (len(matches) > 0):
        #     import pdb
        #     pdb.set_trace()
        # Update track set.
        for track_idx, detection_idx in matches:
            # track_bb = self.tracks[track_idx].get_detection().to_tlbr()
            # detection_bb = detections[detection_idx].to_tlbr()
            # import pdb
            # pdb.set_trace()
            # print("matched_id: ", track_idx, detection_idx)
            # print("OG: ", track_bb)
            # print("DET:", detection_bb)
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
            matched_id = self.tracks[track_idx].track_id

            if (matched_id in self.metrics["id_frames"]):
                self.metrics["id_frames"][matched_id] += 1
            else:
                self.metrics["id_frames"][matched_id] = 1

        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])

        for t in self.tracks:
            if (t.is_deleted()):
                self.metrics["removed"] += 1
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        # self.metric.partial_fit(
        #     np.asarray(features), np.asarray(targets), active_targets)
        self.metric.partial_fit(features, targets, active_targets)

    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            # features = [dets[i].feature for i in detection_indices]
            # targets = [tracks[i].track_id for i in track_indices]
            cost_matrix = self.metric.distance(features, targets)
           # cost_matrix[cost_matrix<0.5] =
            cost_matrix = linear_assignment.gate_cost_neural(
                cost_matrix, nn_margin)
            if (self.use_kalman):
                cost_matrix = linear_assignment.gate_cost_matrix(
                    self.kf, cost_matrix, tracks, dets, track_indices,
                    detection_indices)

            if (self.print_cost_matrix):
                print("Cost Matrix:", cost_matrix)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        # print(confirmed_tracks, "SELF.TRACKS?? CONFIRMED", self.n_init, "OK?")
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)
        # print(matches_a, "Matches")

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)
        matches = matches_a + matches_b
        # For loggging Appearance tracks Ids:
        matched_track_ids = []
        for element in matches_a:
            track_idx = element[0]
            matched_track_ids.append(self.tracks[track_idx].track_id)
        # print("Appeance Matched IDs: ", [matched_track_ids])
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature, detection, use_kalman=self.use_kalman, start_confirmed=self.start_confirmed))
        self._next_id += 1
        self.metrics["new_ids"] += 1
