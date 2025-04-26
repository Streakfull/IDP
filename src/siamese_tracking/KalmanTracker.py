from filterpy.kalman import KalmanFilter
import numpy as np
from pytorchmodels.Tracker import Tracker


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox, use_scores=True):
        """
        Initializes a tracker using the initial bounding box.

        Args:
            bbox: Initial bounding box in [x1, y1, x2, y2, (optional: score, class)] format.
            use_scores: Whether to store the detection confidence score and classification label.
        """

        # Initialize Kalman filter for 7D state: [x1, y1, x2, y2, vx, vy, vw]
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.tracker = Tracker()
        # State transition matrix (models constant velocity)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],  # x1
            [0, 1, 0, 0, 0, 1, 0],  # y1
            [0, 0, 1, 0, 0, 0, 1],  # x2
            [0, 0, 0, 1, 0, 0, 0],  # y2
            [0, 0, 0, 0, 1, 0, 0],  # vx
            [0, 0, 0, 0, 0, 1, 0],  # vy
            [0, 0, 0, 0, 0, 0, 1],  # vw
        ])

        # Measurement matrix (maps state to bounding box measurements)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],  # x1
            [0, 1, 0, 0, 0, 0, 0],  # y1
            [0, 0, 1, 0, 0, 0, 0],  # x2
            [0, 0, 0, 1, 0, 0, 0],  # y2
        ])

        # Measurement noise covariance (noise in detection measurements)
        self.kf.R = np.eye(4) * 1.0  # Tune this based on your detection noise

        # Process noise covariance (uncertainty in state transition)
        self.kf.Q = np.eye(7) * 0.1  # Lower initial values, adjust as needed
        self.kf.Q[4:, 4:] *= 0.01  # Smaller noise for velocity components

        # State covariance matrix (initial uncertainty in state estimate)
        self.kf.P = np.eye(
            7) * 10.  # Higher uncertainty for positions initially
        self.kf.P[4:, 4:] *= 1000.  # Very high uncertainty for velocity

        # Initialize the state with the given bounding box
        # Convert bbox to state vector format
        self.kf.x[:4] = self.tracker.convert_bbox_to_z(bbox)

        # Additional tracker attributes
        self.time_since_update = 0  # Frames since the tracker was last updated
        self.id = KalmanBoxTracker.count  # Assign unique ID to tracker
        KalmanBoxTracker.count += 1
        self.history = []  # Store the history of predictions
        self.hits = 0  # Number of successful updates
        self.hit_streak = 0  # Consecutive updates without missing
        self.age = 0  # Total number of frames the tracker has been alive
        self.n_miss = 0  # Counter for consecutive misses

        # Optionally store detection confidence score and classification label
        if use_scores and len(bbox) >= 6:
            self.score = bbox[4]
            self.classif = bbox[5]
        else:
            self.score = None
            self.classif = None

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self.tracker.convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if ((self.kf.x[6]+self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.tracker.convert_x_to_bbox(
            self.kf.x, self.score, self.classif))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.tracker.convert_x_to_bbox(self.kf.x, self.score, self.classif)
