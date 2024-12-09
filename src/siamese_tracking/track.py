# vim: expandtab:ts=4:sw=4


from siamese_tracking.KalmanTracker import KalmanBoxTracker


class Track:
    def __init__(self, track_id, queue_size=10, feature=None, detection=None,):
        self.track_id = track_id
        self.detection = detection
        self.detection.set_id(self.track_id)
        self.n_miss = 0

        self.features = [detection.feature]
        self.queue_size = queue_size
        if feature is not None:
            self.features.append(feature)
        self.kf = KalmanBoxTracker(self.to_tlbr(), use_scores=False)

    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """

        return self.detection.tlwh.copy()

    def to_tlbr(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.to_tlwh()
        # if (not self.use_kalman):
        #     ret[2:] += ret[:2]
        #     return ret
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def update(self, detection):
        self.features.append(detection.feature)
        self.detection = detection
        detection.set_id(self.track_id)
        if len(self.features) > self.queue_size:
            self.features.pop(0)

    def get_detection(self):
        return self.detection
