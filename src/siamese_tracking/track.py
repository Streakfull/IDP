# vim: expandtab:ts=4:sw=4


from siamese_tracking.KalmanTracker import KalmanBoxTracker
from deep_sort.kalman_filter import KalmanFilter


class Track:
    def __init__(self, track_id, queue_size=10, feature=None, detection=None,
                 mean=0, covariance=0, frame=0

                 ):
        self.track_id = track_id
        self.detection = detection
        self.detection.set_id(self.track_id)
        self.n_miss = 0
        self.mean = mean
        self.covariance = covariance
        self.features = [detection.feature]
        self.frames = [frame]
        self.queue_size = queue_size
        if feature is not None:
            self.features.append(feature)
        self.frame_dist_compare = 50
        # self.kf = KalmanBoxTracker(self.to_tlbr(), use_scores=False)

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

    # def update(self, detection, current_frame):
    #     self.features.append(detection.feature)
    #     self.detection = detection
    #     detection.set_id(self.track_id)
    #     if len(self.features) > self.queue_size:
    #         self.features.pop(0)

    def update(self, detection, current_frame):
        new_feature = detection.feature

        if self.features:  # Check if the queue is not empty
            # Get the most recent frame from the frame list
            top_frame = self.frames[-1]

            # Calculate the frame difference
            frame_difference = abs(current_frame - top_frame)

            if frame_difference < self.frame_dist_compare:
                # Replace the top feature with the new one
                self.features[-1] = new_feature
                self.frames[-1] = current_frame
                # print(
                #     f"Feature replaced, frame difference was {frame_difference}")
            else:
                # Add the new feature and frame to their respective lists
                # print(
                #     f"Feature added, frame difference was {frame_difference}")
                self.features.append(new_feature)
                self.frames.append(current_frame)
        else:
            # If the queue is empty, simply add the new feature and frame
            self.features.append(new_feature)
            self.frames.append(current_frame)
           # print("First feature added.")

        # Maintain queue size
        if len(self.features) > self.queue_size:
            self.features.pop(0)  # Remove the oldest feature
            self.frames.pop(0)    # Remove the corresponding frame

        # Update detection and ID
        self.detection = detection
        detection.set_id(self.track_id)

        def get_detection(self):
            return self.detection

    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)

    def update_kalman(self, kf, detection):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """
        # import pdb
        # pdb.set_trace()
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())

    def to_tlwh_kalman(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr_kalman(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.to_tlwh_kalman()
        # if (not self.use_kalman):
        #     ret[2:] += ret[:2]
        #     return ret
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def append_miss(self):
        self.n_miss += 1
