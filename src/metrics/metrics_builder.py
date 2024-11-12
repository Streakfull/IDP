class Metrics():
    def __init__(self, metrics_config):
        self.metrics_config = metrics_config.split(",")
        self.metrics = []

    def get_metrics(self):
        raise Exception("Metric not supported")
