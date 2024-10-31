import numpy as np
from pytorchmodels.Tracker import Tracker
from pytorchmodels.KalmanBoxTracker import KalmanBoxTracker


class Sort(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.tracker = Tracker()
        self.metrics = {
            "removed": 0,
            "id_frames": {

            },
            "new_ids": 0
        }

    def update(self, dets=np.empty((0, 6))):
        """
        Params:
        dets - a numpy array of detections in the format [[x1,y1,x2,y2,score, classification],[x1,y1,x2,y2, score, classification],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """

        self.frame_count += 1

        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))

        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = self.tracker.associate_detections_to_trackers(
            dets, trks, self.iou_threshold)

        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])
            matched_id = self.trackers[m[1]].id
            if (matched_id in self.metrics["id_frames"]):
                self.metrics["id_frames"][matched_id] += 1
            else:
                self.metrics["id_frames"][matched_id] = 1

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.metrics["new_ids"] += 1
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                # +1 as MOT benchmark requires positive
                ret.append(np.concatenate((d, [trk.id+1])).reshape(1, -1))
            i -= 1
            # remove dead tracklet
            if (trk.time_since_update > self.max_age):
                self.trackers.pop(i)
                self.metrics["removed"] += 1

        if (len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 6))
