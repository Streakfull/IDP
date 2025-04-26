import torchvision.transforms as T
import os
import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from pytorchmodels.jerseyId.strhub.data.module import SceneTextDataModule


data_transforms = {
    'train': {
        'resnet':
            transforms.Compose([
                transforms.RandomGrayscale(),
                transforms.ColorJitter(brightness=.5, hue=.3),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                     0.229, 0.224, 0.225])  # Image Net
                # transforms.Normalize(mean=[0.548, 0.529, 0.539], std=[0.268, 0.280, 0.274]) # Hockey
            ]),
        'vit':
            transforms.Compose([
                transforms.RandomGrayscale(),
                transforms.ColorJitter(brightness=.5, hue=.3),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                     0.229, 0.224, 0.225])  # Image Net
                # transforms.Normalize(mean=[0.548, 0.529, 0.539], std=[0.268, 0.280, 0.274]) # Hockey
            ]),
    },

    'val': {
        'resnet':
            transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                     0.229, 0.224, 0.225])  # ImageNet
                # transforms.Normalize(mean=[0.548, 0.529, 0.539], std=[0.268, 0.280, 0.274]) # Hockey
            ]),
        'vit':
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                     0.229, 0.224, 0.225])  # ImageNet
                # transforms.Normalize(mean=[0.548, 0.529, 0.539], std=[0.268, 0.280, 0.274]) # Hockey
            ])
    },
    'test': {
        'resnet':
        transforms.Compose([  # same as val
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])  # ImageNet
            # transforms.Normalize(mean=[0.548, 0.529, 0.539], std=[0.268, 0.280, 0.274]) # Hockey
        ]),
        'vit':
        transforms.Compose([  # same as val
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])  # ImageNet
            # transforms.Normalize(mean=[0.548, 0.529, 0.539], std=[0.268, 0.280, 0.274]) # Hockey
        ]),
    }
}


class JerseyDataset(Dataset):
    def __init__(self, root_dir,  split='train', model_type='resnet', transform=None):
        """
        Args:
            root_dir (str): Path to the dataset root (should contain 'train' and 'test' folders).
            split (str): One of ['train', 'val', 'test'].
            model_type (str): One of ['resnet', 'vit'] to apply corresponding transformations.
            transform (callable, optional): Optional transform to be applied to each image.
        """
        assert split in ['train', 'val',
                         'test', 'trainTest'], "split must be 'train', 'val', or 'test'"
        assert model_type in [
            'resnet', 'vit'], "model_type must be 'resnet' or 'vit'"
        self.use_combined = split == "train"
        self.root_dir = root_dir

        self.gt_path = os.path.join(self.root_dir, f'{split}_gt.json')
        self.transform = transform if transform else data_transforms[split][model_type]

        if self.use_combined:
            # If 'trainTest' is selected, merge train and test splits
            train_samples = self._get_samples('train')
            test_samples = self._get_samples('test')
            self.samples = train_samples + test_samples
        else:
            # Otherwise, load data for the specified split (train, val, test)
            self.samples = self._get_samples(split)

    @staticmethod
    def test_transform():
        return data_transforms['test']["resnet"]

    def _get_samples(self, split):
        """
        Helper function to get the samples for a specific split (train, test, val).

        Args:
            split (str): The split to fetch ('train', 'test', 'val').

        Returns:
            List: List of samples (image path and label).
        """
        assert split in ['train', 'test',
                         'val'], "split must be 'train', 'test', or 'val'"

        gt_path = os.path.join(self.root_dir, split, f'{split}_gt.json')
        with open(gt_path, 'r') as f:
            labels = json.load(f)

        # Exclude tracklets with jersey number -1
        tracklets = [tid for tid, label in labels.items() if label != -1]
        image_dir = os.path.join(self.root_dir, split, 'images')
        samples = []
        for tracklet_id in tracklets:
            tracklet_folder = os.path.join(image_dir, str(tracklet_id))
            if os.path.exists(tracklet_folder):
                for img_name in sorted(os.listdir(tracklet_folder)):
                    if img_name.endswith('.jpg'):
                        img_path = os.path.join(tracklet_folder, img_name)
                        samples.append((img_path, labels[tracklet_id]))

        return samples

    def __len__(self):
        # return 200
        return len(self.samples)

    def get_digit_labels(self, label):
        if label < 10:
            return label, 10
        else:
            return label // 10, label % 10

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        # raw_img = image.copy()
       # raw_img = self.transformp(image)
        # kp = image.copy()
        # kp = self.transform_kp(kp)
        if self.transform:
            image = self.transform(image)
        digit1, digit2 = self.get_digit_labels(label)

        return {"image": image,
                "label": torch.tensor(label, dtype=torch.long),
                "digit1": torch.tensor(digit1, dtype=torch.long),
                "digit2": torch.tensor(digit2, dtype=torch.long),
                # "kp": kp,
                # "imgp": raw_img

                }

    @staticmethod
    def move_batch_to_device(batch, device):
        pass
        # batch["image"] = batch["image"].to(device)
        # batch["label"] = batch["label"].to(device)
        # batch["digit1"] = batch["digit1"].to(device)
        # batch["digit2"] = batch["digit2"].to(device)
        # batch["kp"] = batch['kp'].to(device)
        # batch["imgp"] = batch["imgp"].to(device)

    def _load_labels(self, gt_path):
        """Helper method to load ground truth labels from a json file."""
        with open(gt_path, 'r') as f:
            labels = json.load(f)
        return labels

    def load_frame_crops(self, det, frame_path):
        x_img = Image.open(frame_path).convert("RGB")
        boxes = det.boxes.xywh.cpu()
        x_crops = []
        for bb in boxes:
            z = torch.zeros(2)
            box = torch.cat((z, bb))
         #   crop = self.crop_image(x_img, box.numpy(), self.crop_size)
            crop = self.crop_image(x_img, box.numpy())
            crop = self.transform(crop)
            x_crops.append(crop)
        x_img = self.transformf(x_img)

        if len(boxes) == 0:
            # Return empty tensors if no detections are found
            return torch.empty(0).to("cuda:0"),  x_img.to("cuda:0")
        x_crops = torch.stack(x_crops)

        return x_crops.to("cuda:0"), x_img.to("cuda:0")

    @staticmethod
    def crop_image(img, meta):
        """
        Extracts the exact bounding box from the image and then applies padding to match crop_size.

        Args:
            img (PIL.Image): Input image.
            meta (tuple): Metadata containing bounding box (x, y, w, h).

        Returns:
            PIL.Image: Cropped and padded image of size (225, 225).
        """
        x, y, w, h = meta[2], meta[3], meta[4], meta[5]

        # Extract the bounding box region
        cropped_img = img.crop((x, y, x + w, y + h))

        # Pad and resize to crop_size
        return cropped_img

    @staticmethod
    def load_frame_crops_det(det, frame_path):
        x_img = Image.open(frame_path).convert("RGB")
        x_crops = []
        boxes = [torch.Tensor(d.get_xywh()) for d in det]
        for bb in boxes:
            z = torch.zeros(2)
            box = torch.cat((z, bb))
         #   crop = self.crop_image(x_img, box.numpy(), self.crop_size)
            crop = JerseyDataset.crop_image(x_img, box.numpy())
            crop = JerseyDataset.test_transform()(crop)
            x_crops.append(crop)

        if len(boxes) == 0:
            # Return empty tensors if no detections are found
            return torch.empty(0).to("cuda:0")
        x_crops = torch.stack(x_crops).to("cuda:0")

        return x_crops.to("cuda:0")
