class Track:

    def __init__(self, track_id, detection=None):
        self.detection = detection
        self.track_id = track_id

    def get_detection(self):
        return self.detection

    def get_id(self):
        return self.track_id

    def set_detection(self, detection):
        self.detection = detection
