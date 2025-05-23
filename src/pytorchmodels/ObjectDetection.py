from types import MethodType
import cv2
import torch
import cvzone
from sort import *
from tqdm.notebook import tqdm
from ultralytics import YOLO


class ObjectDetection(torch.nn.Module):
    def __init__(self, capture) -> None:
        super().__init__()
        self.capture = capture
        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.model.names

    def load_model(self):
        model = YOLO("yolo11x.pt")
        model.fuse()
        return model

    def predict(self, img):
        results = self.model(img)
        # import pdb
        # pdb.set_trace()
        return results

    def plot_boxes(self, results, img):
        for box in results[0].boxes:

            x1, y1, x2, y2 = box.xyxy[0]
            cls, conf = box.cls, box.conf
            x1, y1, x2, y2, conf, cls = int(x1), int(y1), int(
                x2), int(y2), round(float(conf), 2), int(cls)
            w, h = x2-x1, y2-y1
            current_class = self.CLASS_NAMES_DICT[cls]
            if (conf > 0.01):
                cvzone.putTextRect(
                    img, f'cls: {current_class}', (x1, y1), scale=1, thickness=1, colorR=(0, 0, 255))

                cvzone.cornerRect(img, (x1, y1, w, h), l=9,
                                  rt=1, colorR=(255, 0, 255))
        return img

    def process_video(self, video, write_path="./logs/outputLive/"):
        cap = cv2.VideoCapture(video)
        # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames = 10
        frame = 0
        with tqdm(total=total_frames-1, desc="Processing frames", unit="frame") as pbar:
            while True:
                _, img = cap.read()
                results = self.predict(img)
                frames = self.plot_boxes(results, img)
                # cv2.imshow('Image', frames)

                cv2.imwrite(f"{write_path}/frame_{frame}.jpg", frames)
                frame += 1
                pbar.update(1)
                if (frame >= total_frames):
                    break
                # if (cv2.waitKey(1) == ord('q') or frame >= total_frames):
                #     break
            cap.release()
            # cv2.destroyAllWindows()

    def process_video_stored(self, video, write_path="../logs/outputVideo"):
        cap = cv2.VideoCapture(video)
        assert cap.isOpened()
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        with tqdm(total=total_frames-1, desc="Processing frames", unit="frame") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                result = self.predict(frame)
                result.save(
                    save_dir=f"{write_path}/frame", exist_ok=False)
                frame_count += 1
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                pbar.update(1)
            cap.release()
            cv2.destroyAllWindows()

    def get_full_pred(self, det):
        try:
            boxes = det[0].boxes.xyxy
        except:
            import pdb
            pdb.set_trace()

        cls = det[0].boxes.cls.unsqueeze(1)
        conf = det[0].boxes.conf.unsqueeze(1)
        detections = torch.cat((boxes, conf, cls), dim=1)
        return detections
