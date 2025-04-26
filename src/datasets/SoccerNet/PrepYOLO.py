import os
import json
import shutil
from tqdm import tqdm


class ReorganizeDataset:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

    def preprocess_label(self, label):
        """Format label: lowercase and replace spaces with underscores."""
        return label.lower().replace(" ", "_")

    def process_split(self, split):
        """Reorganize images and labels by split."""
        print(f"Processing split: {split}")
        split_input_dir = os.path.join(self.input_path, split)
        split_output_dir = os.path.join(self.output_path, split)

        # Create split output directory
        os.makedirs(split_output_dir, exist_ok=True)

        # Get list of all image files in the split
        image_dir = os.path.join(split_input_dir, "images")
        label_dir = os.path.join(split_input_dir, "labels")
        images = [f for f in os.listdir(
            image_dir) if f.endswith((".png", ".jpg"))]

        for image_file in tqdm(images, desc=f"Processing {split} images"):
            # Extract the label from the JSON file
            label_file = os.path.join(
                label_dir, f"{os.path.splitext(image_file)[0]}.json")
            if not os.path.exists(label_file):
                print(f"Warning: Label file missing for {image_file}")
                continue

            with open(label_file, "r") as lf:
                annotation = json.load(lf)
                label = self.preprocess_label(annotation["label"])

            # Create the label folder in the output structure
            label_output_dir = os.path.join(split_output_dir, label)
            os.makedirs(label_output_dir, exist_ok=True)

            # Copy the image to the new folder
            source_image_path = os.path.join(image_dir, image_file)
            target_image_path = os.path.join(label_output_dir, image_file)
            shutil.copy(source_image_path, target_image_path)

    def reorganize(self):
        """Main function to reorganize the dataset."""
        for split in ["train", "valid", "test"]:
            self.process_split(split)


if __name__ == "__main__":
    input_dataset_path = "./raw_dataset/soccerNetV2/preprocessed"
    output_dataset_path = "./raw_dataset/soccerNetV2/preprocessed/YOLO"

    reorganizer = ReorganizeDataset(input_dataset_path, output_dataset_path)
    reorganizer.reorganize()
