import cv2
import torch
import cvzone
from sort import *


class ObjectDetection(torch.nn.Module):
    def __init__(self, capture) -> None:
        super().__init__()
        self.capture = capture
        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.model.names

    def load_model(self):
        model = torch.hub.load('ultralytics/yolov5',
                               'yolov5n', pretrained=True)
        return model

    def predict(self, img):
        results = self.model(img)
        return results
