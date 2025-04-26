from types import MethodType

from ultralytics import YOLO
from pytorchmodels.ObjectDetection import ObjectDetection
import cv2
import torch
import cvzone
from sort import *
from tqdm.notebook import tqdm
from deep_sort.Detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import nn_matching
from deep_sort.nms import non_max_suppression
import numpy as np
import uuid
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from PIL import Image

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
        0.229, 0.224, 0.225]),  # ImageNet normalization
])

preprocess_resnet = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


preprocess_vit = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def _predict_once(self, x, profile=False, visualize=False, embed=None):
    y, dt, embeddings = [], [], []  # outputs
    for m in self.model:
        if m.f != -1:  # if not from previous layer
            x = y[m.f] if isinstance(m.f, int) else [
                x if j == -1 else y[j] for j in m.f]  # from earlier layers
        if profile:
            self._profile_one_layer(m, x, dt)
        x = m(x)  # run
        y.append(x if m.i in self.save else None)  # save output
        # if visualize:
        #     feature_visualization(x, m.type, m.i, save_dir=visualize)

        # Change this so that it returns the feature maps without any change
        if embed and m.i in embed:
            embeddings.append(x)  # flatten
            if m.i == max(embed):
                return embeddings
    return x


class DeepSortObjectTracking(ObjectDetection):

    def __init__(self, capture, write_path) -> None:
        super().__init__(capture)
        self.min_confidence = 0.25
        self.max_cosine_distance = 0.2
        self.nn_budget = None
        self.write_path = write_path
        self.frame_count = 0
        self.embedding_model = self.load_embedding_model()
        self.vgg_extracor = self.load_vgg()
        self.resnet_extractor = self.load_resnet()
        self.vit_extractor = self.load_vit()
        self.vgg_extracor.eval()

    # def load_model(self):
    #     model = super().load_model()
    #     self.model = model
    #     self.model.model._predict_once = MethodType(_predict_once, model.model)
    #     return model

    def load_vgg(self):
        vgg16 = models.vgg16(pretrained=True)
        vgg16.eval()
        vgg16_features = vgg16.features.to("cuda:0")
        return vgg16_features

    def load_resnet(self):
        resnet50 = models.resnet50(pretrained=True)
        feature_extractor = nn.Sequential(*list(resnet50.children())[:-1])
        resnet_features = feature_extractor.to("cuda:0")
        return resnet_features

    def load_vit(self):
        model = models.vit_b_16(pretrained=True)
        model.eval()  # Set to evaluation mode
        model.heads = torch.nn.Identity()
        model = model.to("cuda:0")
        return model

    def load_embedding_model(self):
        model = YOLO("yolo11n.pt")
        model.fuse()
        return model

    def process_video(self, video, write_path="./logs/outputLive/"):
        cap = cv2.VideoCapture(video)
        # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames = 2
        frame = 0
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", self.max_cosine_distance, None)
        tracker = Tracker(metric)
       # results = []
        with tqdm(total=total_frames-1, desc="Processing frames", unit="frame") as pbar:
            while True:
                _, img = cap.read()
                det = self.predict(img)
                # if frame > 0:
                #     import pdb
                #     pdb.set_trace()
                detections = self.get_detections_objects(det, img)
                detections = [
                    d for d in detections if d.confidence >= self.min_confidence]
                tracker.predict()
                tracker.update(detections)
                results = []
                for track in tracker.tracks:
                    if (not track.is_confirmed() or track.time_since_update > 1) and (frame > tracker.n_init):
                        continue

                    bbox = track.to_tlbr()
                    conf = track.get_detection().get_conf()
                    cls = track.get_detection().get_cls()

                    cls = cls.cpu().numpy()
                    cls = np.array([cls])
                    conf = np.array([conf])
                    id = np.array([track.track_id])
                    result = np.concatenate((bbox, conf, cls, id))
                    results.append(result)
                frames = self.plot_boxes(results, img)

                cv2.imwrite(f"{write_path}/frame_{frame}.jpg", frames)
                frame += 1
                self.frame_count += 1
                pbar.update(1)
                if (cv2.waitKey(1) == ord('q')):
                    break
                if frame == total_frames:
                    break
            cap.release()
            cv2.destroyAllWindows()
        return tracker.metrics

    def plot_boxes(self, results, img):
        for box in results:

            x1, y1, x2, y2, conf, cls, id = box
            x1, y1, x2, y2, conf, cls, id = int(x1), int(y1), int(
                x2), int(y2), round(float(conf), 2), int(cls), int(id)
            w, h = x2-x1, y2-y1
            current_class = self.CLASS_NAMES_DICT[cls]

            if (conf > 0.25):
                cvzone.putTextRect(
                    img, f'{current_class[0]}, {id}, {conf}', (x1, y1), scale=1, thickness=1, colorR=(0, 0, 255))

                cvzone.cornerRect(img, (x1, y1, w, h), l=9,
                                  rt=1, colorR=(255, 0, 255))
        return img

    def get_detections_objects(self, det, frame):
        results = self.get_full_pred(det)
        # yolo_features = self.get_yolo_features(results, frame)
        # features = self.get_vgg_features(results, frame)
        features = self.get_yolo_features(results, frame)
        # features = self.get_resnet_features(results, frame)
        # features = self.get_vit_features(results, frame)
        objects = list(map(Detection, results))
        for i in range(len(features)):
            feat = features[i]
            # feat_b = features_b[i]
            # full_feat = (feat, feat_b)
            detection = objects[i]
            detection.set_feature(feat)
        return objects

    def get_crops(self, res, frame):
        crop_objects = []
        for box in res:
            crop_obj = frame[int(box[1]): int(
                box[3]), int(box[0]): int(box[2])]
            crop_objects.append(crop_obj)
        return crop_objects

    def get_yolo_features(self, res, frame):
        crop_objects = self.get_crops(res, frame)

        # embeddings = self.embedding_model.embed(crop_objects)
        # embeddings = self.embedding_model.embed(frame)
        embeddings = self.embedding_model(frame)
        import pdb
        pdb.set_trace()
        # return embeddings
        return []

    def get_vgg_features(self, res, frame):
        crop_objects = self.get_crops(res, frame)
        for i in range(len(crop_objects)):
            crop_objects[i] = Image.fromarray(crop_objects[i])
        crop_t = []
        for i in range(len(crop_objects)):
            crop_t.append(preprocess(crop_objects[i]))
            # crop_t.append(crop_objects[i])
        crop_t = torch.stack(crop_t)
        # import pdb
        # pdb.set_trace()
        crop_t = torch.tensor(crop_t)
        with torch.no_grad():
            embeddings = self.vgg_extracor(crop_t.to("cuda:0"))
            embeddings = embeddings.flatten(1)
        return embeddings

    def get_resnet_features(self, res, frame):
        crop_objects = self.get_crops(res, frame)
        for i in range(len(crop_objects)):
            crop_objects[i] = Image.fromarray(crop_objects[i])
        crop_t = []
        for i in range(len(crop_objects)):
            crop_t.append(preprocess_resnet(crop_objects[i]))
            # crop_t.append(crop_objects[i])
        crop_t = torch.stack(crop_t)
        # import pdb
        # pdb.set_trace()
        crop_t = torch.tensor(crop_t)
        with torch.no_grad():
            embeddings = self.resnet_extractor(crop_t.to("cuda:0"))
            embeddings = embeddings.flatten(1)
        return embeddings

    def get_vit_features(self, res, frame):
        crop_objects = self.get_crops(res, frame)
        for i in range(len(crop_objects)):
            crop_objects[i] = Image.fromarray(crop_objects[i])
        crop_t = []
        for i in range(len(crop_objects)):
            crop_t.append(preprocess_resnet(crop_objects[i]))
            # crop_t.append(crop_objects[i])
        crop_t = torch.stack(crop_t)
        # import pdb
        # pdb.set_trace()
        crop_t = torch.tensor(crop_t)
        with torch.no_grad():
            embeddings = self.vit_extractor(crop_t.to("cuda:0"))
            embeddings = embeddings.flatten(1)
        return embeddings
