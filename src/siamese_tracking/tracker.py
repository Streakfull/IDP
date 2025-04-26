import pdb
import numpy as np
import torch
from deep_sort import linear_assignment

from deep_sort.iou_matching import iou_cost
from siamese_tracking.Detection import Detection
from siamese_tracking.track import Track
from deep_sort import iou_matching
from lutils.general import find_object_by_track_id
from deep_sort.kalman_filter import KalmanFilter


nn_margin = 0.1


class Tracker:

    def __init__(self, metric, max_iou_distance=0.9, use_enhance=False, track_queue_size=10):
        self.tracks = []
        self.tracken_map = {}
        self._next_id = 1
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.use_enhance = use_enhance
        self.track_queue_size = track_queue_size
        self.kf = KalmanFilter()
        self.last_full_tracks = 0
        self.n_pred = 0
        self.current_frame = 0

    def update(self, detections, current_frame):
        # Get cost matrix between detections and all tracks
        self.current_frame = current_frame
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)
        if (len(matches) > 0):
            self.adjust_ids(detections, matches, current_frame)

        for det_idx in unmatched_detections:
            self._initiate_track(detections[det_idx], current_frame)

        active_targets = [t.track_id for t in self.tracks]
        features, targets = [], []
        for track in self.tracks:
            features += track.features
            targets += [track.track_id for _ in track.features]
        self.metric.partial_fit(features, targets, active_targets)
        for track in unmatched_tracks:
            self.tracks[track].append_miss()

        if (self.use_enhance):
            pred_det = self.update_track_enhance_map(
                detections, unmatched_tracks, current_frame)

            return pred_det

        # self.tracks = [track for track in self.tracks if track.n_miss < 30]

        return None

    def update_track_enhance_map(self, detections_t, unmatched_tracks_t, current_frame):
        # Matches ---> update previous map
        for detection in detections_t:
            track = self.tracken_map.get(detection.id, None)
            if (track is None):
                track = find_object_by_track_id(self.tracks, detection.id)
                self.tracken_map[track.track_id] = track
            track.n_miss = 0
            # track.kf.update(detection.to_tlbr())
            track.predict(self.kf)
            track.update_kalman(self.kf, detection)

        predicted_det = []
        for tr in unmatched_tracks_t:
            track_id = self.tracks[tr].track_id
            if not track_id in self.tracken_map.keys():
                continue
            track = self.tracken_map[track_id]
            track.n_miss += 1
            if (track.n_miss > 10):
                del self.tracken_map[track_id]
            if (track.n_miss > 3):
                continue

            pred = track.to_tlbr()
            sc = [0.5, 0]
            pred = np.append(pred, sc)
            det = Detection(torch.Tensor(pred))
            det.set_id(track_id)
            predicted_det.append(det)

        if (len(predicted_det) == 0):
            print("Resetting last full tracks at frame: ",
                  current_frame, "tracks: ", len(detections_t))
            self.last_full_tracks = len(detections_t)
            self.n_pred = 0
            return []

        if (len(detections_t) >= self.last_full_tracks and self.n_pred < 10):
            predicted_det = []

        if (len(predicted_det) > 0):
            self.n_pred += 1

        return predicted_det

    def _match(self, detections):
        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_neural(
                cost_matrix, nn_margin)
            # iou_mask = np.ones(cost_matrix.shape)
            # iou_matches, unmatched_tracks_iou, unmatched_detections = \
            #     linear_assignment.min_cost_matching(
            #         iou_matching.iou_cost, self.max_iou_distance, self.tracks,
            #         detections, track_indices, detection_indices=None
            #     )
            # for i, track_idx in enumerate(track_indices):
            #     for j, det_idx in enumerate(detection_indices):
            #         if (track_idx, det_idx) not in iou_matches:
            #             iou_mask[i, j] = 1e5  # Mask non-IOU matches
            # cost_matrix = cost_matrix * iou_mask

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
        print("Frame: ", self.current_frame, ",", "Appeance Matched IDs: ", len(
            matched_track_ids), [matched_track_ids])
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))

        return matches, unmatched_tracks, unmatched_detections

    def _match2(self, detections):
        # Iou>appearance>Iou
        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_neural(
                cost_matrix, nn_margin)

            iou_mask = np.ones(cost_matrix.shape)
            iou_matches, unmatched_tracks_iou, unmatched_detections = \
                linear_assignment.min_cost_matching(
                    iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                    detections, track_indices, detection_indices=None
                )
            for i, track_idx in enumerate(track_indices):
                for j, det_idx in enumerate(detection_indices):
                    if (track_idx, det_idx) not in iou_matches:
                        iou_mask[i, j] = 1e5  # Mask non-IOU matches
            cost_matrix = cost_matrix * iou_mask

            return cost_matrix

        track_indices = [i for i, t in enumerate(self.tracks)]

        # Step 1: IoU matching only for tracks with n_miss < 3
        valid_iou_tracks = [
            i for i in track_indices if self.tracks[i].n_miss < 30]
        matches_iou1, unmatched_tracks_iou1, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, valid_iou_tracks, detection_indices=None
            )
        um1 = unmatched_detections

        # Step 2: Appearance-based matching (for tracks unmatched in Step 1)

        matches_appearance, unmatched_tracks_appearance, unmatched_detections = \
            linear_assignment.matching_cascade_no_max_age(
                gated_metric, self.metric.matching_threshold,
                self.tracks, detections, track_indices=unmatched_tracks_iou1,
                detection_indices=unmatched_detections, cascade_depth=5000)
        # if (self.current_frame == 30):
        #     import pdb
        #     pdb.set_trace()

        um2 = unmatched_detections
        matches_iou2, unmatched_tracks_final, unmatched_detections = \
            linear_assignment.min_cost_matching(iou_matching.iou_cost, self.max_iou_distance,
                                                self.tracks, detections, unmatched_tracks_appearance, unmatched_detections)

        um3 = unmatched_detections
        # if (len(unmatched_detections) >= 1 and self.current_frame >= 50):
        #     pdb.set_trace()
        # Merge matches

        matches = matches_iou1 + matches_appearance + matches_iou2
        matched_track_ids = [
            self.tracks[element[0]].track_id for element in matches]
        unmatched_track_ids = [
            self.tracks[element].track_id for element in unmatched_tracks_final]

        print(f"Frame: {self.current_frame}, "
              f"IoU1 Matches: {len(matches_iou1)}, "
              f"Appearance Matches: {len(matches_appearance)}, "
              f"IoU2 Matches: {len(matches_iou2)}, "
              f"Total Matched Track IDs: {len(matched_track_ids)}, {matched_track_ids}",
              f"Total Unmatched: {len(unmatched_detections)}",
              f"Total Unmatched Tracks: {unmatched_track_ids}",
              )

        return matches, unmatched_tracks_final, unmatched_detections

    def _initiate_track(self, detection, frame):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        track = Track(
            self._next_id,
            detection=detection,
            queue_size=self.track_queue_size,
            mean=mean,
            covariance=covariance,
            frame=frame
        )

        self.tracks.append(track)
        self._next_id += 1

    def adjust_ids(self, detections, matches, current_frame):
        for match in matches:
            track_idx, det_idx = match
            track = self.tracks[track_idx]
            detection = detections[det_idx]
            track.update(detection, current_frame)
