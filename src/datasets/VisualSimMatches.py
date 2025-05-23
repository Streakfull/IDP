import os
from torch.utils.data.dataset import Dataset
import numpy as np
from PIL import Image
from datasets.base_dataset import BaseDataSet
from torchvision import transforms
from einops import rearrange
from blocks.pointnet.utils import pc_normalize


class VisualSimMatches(BaseDataSet):
    def __init__(self, dataset_options, sim_matches_options):
        super().__init__(
            dataset_options, sim_matches_options)
        self.dataset_paths = [
            # "./raw_dataset/frame_pairs_far",
            # "./raw_dataset/frame_pairs_full_clipped_150_30_fixed",
            # "./raw_dataset/frame_pairs_full"
            # "./raw_dataset/75k-close-far-norm"
            "./raw_dataset/latestv3/full/samples",
            "./raw_dataset/latestv3/full/clipped_text",
            "./raw_dataset/latestv3/full/clipped_manual_2",
        ]
        self.items = self.get_items()
        self.transform = transforms.Compose([
            # Resize to 224x224 (or your target size)
            transforms.Resize((224, 224)),
            transforms.ToTensor(),          # Convert image to tensor
            # Normalize for pre-trained models
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])

    # def get_items(self):
    #     items = os.listdir(self.dataset_path)
    #     return items

    # def __getitem__(self, index):
    #     # Get the item corresponding to the given index
    #     item_name = self.items[index]

    #     # Construct the path to the image and the associated data
    #     image_path_x1 = os.path.join(self.dataset_path, item_name, 'x1.jpg')
    #     image_path_x2 = os.path.join(self.dataset_path, item_name, 'x2.jpg')
    #     feature_path_x1 = os.path.join(self.dataset_path, item_name, 'x1.txt')
    #     feature_path_x2 = os.path.join(self.dataset_path, item_name, 'x2.txt')

    #     # Load images using PIL
    #     x1_img = Image.open(image_path_x1).convert('RGB')
    #     x2_img = Image.open(image_path_x2).convert('RGB')

    #     # Apply transformations to the images
    #     x1_img = self.transform(x1_img)
    #     x2_img = self.transform(x2_img)

    #     # Load feature vectors (e.g., numpy arrays)
    #     # x1f = np.loadtxt(feature_path_x1)[5:]
    #     # x2f = np.loadtxt(feature_path_x2)[5:]
    #     x1f = np.loadtxt(feature_path_x1)
    #     x2f = np.loadtxt(feature_path_x2)
    #     # Convert the features to tensors
    #     # x1f = torch.tensor(x1f, dtype=torch.float32)
    #     # x2f = torch.tensor(x2f, dtype=torch.float32)

    #     tgt = int(item_name.split("-")[-1].split(".")[0])

    #     # Return a dictionary with the images and features
    #     return {
    #         "x1_img": x1_img,
    #         "x2_img": x2_img,
    #         "x1f": x1f,
    #         "x2f": x2f,
    #         "label": tgt,
    #         "path": item_name
    #     }

    def get_items(self):
        items = []
        for base_path in self.dataset_paths:
            dir_items = os.listdir(base_path)
            items.extend([(base_path, item) for item in dir_items])
        return items

    def __getitem__(self, index):
        # Get the base path and item name for the given index
        base_path, item_name = self.items[index]

        # Construct paths to the images and features
        image_path_x1 = os.path.join(base_path, item_name, 'x1.jpg')
        image_path_x2 = os.path.join(base_path, item_name, 'x2.jpg')
        feature_path_x1_f = os.path.join(base_path, item_name, 'x1f.txt')
        feature_path_x1 = os.path.join(base_path, item_name, 'x1.txt')
        feature_path_x2_f = os.path.join(base_path, item_name, 'x2f.txt')
        feature_path_x2 = os.path.join(base_path, item_name, 'x2.txt')
        feature_path_x1 = feature_path_x1_f if os.path.exists(
            feature_path_x1_f) else feature_path_x1
        feature_path_x2 = feature_path_x2_f if os.path.exists(
            feature_path_x2_f) else feature_path_x2
        x1k, x2k = np.zeros(768), np.zeros(768)
        kp_found = False
        keypoints_x1 = os.path.join(base_path, item_name, 'x1k.txt')
        keypoints_x2 = os.path.join(base_path, item_name, 'x2k.txt')
        if os.path.exists(keypoints_x1) and os.path.exists(keypoints_x2):
            kp_found = True
            x1k = np.loadtxt(keypoints_x1)
            x2k = np.loadtxt(keypoints_x2)
        else:
            print("Not found", base_path, item_name)

        # Load images using PIL
        x1_img = Image.open(image_path_x1).convert('RGB')
        x2_img = Image.open(image_path_x2).convert('RGB')

        # Apply transformations to the images
        x1_img = self.transform(x1_img)
        x2_img = self.transform(x2_img)

        # Load feature vectors
        x1f = np.loadtxt(feature_path_x1)
        x2f = np.loadtxt(feature_path_x2)

        # x1k = self.preprocess_pc(x1k)
        # x2k = self.preprocess_pc(x2k)
        # Extract the label from the item name
        tgt = int(item_name.split("-")[-1].split(".")[0])

        # Return a dictionary with the images, features, and label
        return {
            "x1_img": x1_img,
            "x2_img": x2_img,
            "x1f": x1f,
            "x2f": x2f,
            "x1k": x1k,
            "x2k": x2k,
            "label": tgt,
            "path": os.path.join(base_path, item_name),
            "x1_path": image_path_x1,
            "x2_path": image_path_x2
        }

    @staticmethod
    def move_batch_to_device(batch, device):
        """
        Moves each element of the batch (images and features) to the specified device.
        """
        # Move the images and feature tensors to the specified device (e.g., 'cuda:0' or 'cpu')
        batch["x1_img"] = batch["x1_img"].to(device)
        batch["x2_img"] = batch["x2_img"].to(device)
        batch["x1f"] = batch["x1f"].float().to(device)
        batch["x2f"] = batch["x2f"].float().to(device)
        batch["label"] = batch["label"].long().to(device)
        batch["x1k"] = batch["x1k"].float().to(device)
        batch["x2k"] = batch["x2k"].float().to(device)

        return batch

    def preprocess_pc(self, pc):
        # pc = rearrange(pc, "(bs w)->bs w", bs=17, w=3)
        # pc = pc[:, :2]
        # norm = pc_normalize(pc)
        # pc = np.concatenate((pc, norm), axis=1)
        # pc = rearrange(pc, "N D -> D N")
        return pc

    # def __len__(self):
    #     return 50_000
