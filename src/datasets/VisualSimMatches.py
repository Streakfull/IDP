import os
from torch.utils.data.dataset import Dataset
import numpy as np
from PIL import Image
from datasets.base_dataset import BaseDataSet
from torchvision import transforms


class VisualSimMatches(BaseDataSet):
    def __init__(self, dataset_options, sim_matches_options):
        super().__init__(
            dataset_options, sim_matches_options)
        self.items = self.get_items()
        self.transform = transforms.Compose([
            # Resize to 224x224 (or your target size)
            transforms.Resize((224, 224)),
            transforms.ToTensor(),          # Convert image to tensor
            # Normalize for pre-trained models
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])

    def get_items(self):
        items = os.listdir(self.dataset_path)
        return items

    def __getitem__(self, index):
        # Get the item corresponding to the given index
        item_name = self.items[index]

        # Construct the path to the image and the associated data
        image_path_x1 = os.path.join(self.dataset_path, item_name, 'x1.jpg')
        image_path_x2 = os.path.join(self.dataset_path, item_name, 'x2.jpg')
        feature_path_x1 = os.path.join(self.dataset_path, item_name, 'x1.txt')
        feature_path_x2 = os.path.join(self.dataset_path, item_name, 'x2.txt')

        # Load images using PIL
        x1_img = Image.open(image_path_x1).convert('RGB')
        x2_img = Image.open(image_path_x2).convert('RGB')

        # Apply transformations to the images
        x1_img = self.transform(x1_img)
        x2_img = self.transform(x2_img)

        # Load feature vectors (e.g., numpy arrays)
        x1f = np.loadtxt(feature_path_x1)[5:]
        x2f = np.loadtxt(feature_path_x2)[5:]

        # Convert the features to tensors
        # x1f = torch.tensor(x1f, dtype=torch.float32)
        # x2f = torch.tensor(x2f, dtype=torch.float32)

        tgt = int(item_name.split("-")[-1].split(".")[0])

        # Return a dictionary with the images and features
        return {
            "x1_img": x1_img,
            "x2_img": x2_img,
            "x1_f": x1f,
            "x2_f": x2f,
            "label": tgt
        }

    @staticmethod
    def move_batch_to_device(batch, device):
        # Move each element of the batch (images and features) to the specified device
        x1_img, x2_img, x1f, x2f = batch["x1_img"], batch["x2_img"], batch["x1_features"], batch["x2_features"]
        x1_img = x1_img.to(device)
        x2_img = x2_img.to(device)
        x1f = x1f.to(device)
        x2f = x2f.to(device)

        return {"x1_img": x1_img, "x2_img": x2_img, "x1_features": x1f, "x2_features": x2f}
