from PIL import Image
import pandas as pd
import tqdm
import tqdm
from torchvision import transforms
from collections import Counter
import matplotlib.pyplot as plt
import os
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve
import pdb
import torch
import numpy as np
from training.ModelTrainer_reid_latest import ModelTrainer
from datasets.SimMatches import SimMatches
from datasets.VisualSimMatches import VisualSimMatches
from lutils.general import seed_all
import json
seed_all(111)
x = torch.cuda.mem_get_info()
print(x)


class JerseyEval():
    def __init__(self, output_folder="preds",
                 bb_file="../logs/eval/custom-plz/try2-ball-filter/direct.txt",
                 frames_folder="../logs/benchmarks/clip_1/raw_frames"

                 ):
        self.trainer = ModelTrainer(dataset_type=VisualSimMatches,
                                    options={"tdm_notebook": True})
        dataset = self.trainer.data_loader_handler.dataset
        print("Dataset length: ", len(dataset))
        torch.cuda.empty_cache()
        print(torch.cuda.mem_get_info())

        self.model = self.trainer.model
        self.output_folder = output_folder  # Folder for predictions & PR curves
        self.output_file = os.path.join(
            self.output_folder, "predictions.json")  # Store predictions
        self.pr_values_file = os.path.join(
            self.output_folder, "pr_values.txt")  # Store PR values
        self.accuracy_file = os.path.join(self.output_folder, "accuracy.txt")
        self.gt_test_path = "./raw_dataset/soccernet-jersey/raw/jersey-2023/test/test_gt.json"
        self.gt_train_path = "./raw_dataset/soccernet-jersey/raw/jersey-2023/train/train_gt.json"
        self.bb_file = bb_file
        self.frames_folder = frames_folder
        # Folder to save cropped images
        self.crops_folder = os.path.join(self.output_folder, "custom_crops")

        # Ensure output folders exist
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.crops_folder, exist_ok=True)

        # Clear existing files if they exist
        # for file in [self.output_file, self.pr_values_file]:
        #     if os.path.exists(file):
        #         os.remove(file)
      # HI2

    def compute_class_distribution(self):
        """
        Compute and plot class distribution from the ground truth data in the test and train JSON files.
        """
        # Load the ground truth data
        with open(self.gt_test_path, "r") as f_test, open(self.gt_train_path, "r") as f_train:
            gt_test = json.load(f_test)
            gt_train = json.load(f_train)

        # Combine all ground truth labels from both datasets
        all_labels = []

        # Assuming the GT data is a dictionary with keys as the sample id and values as the class label
        all_labels.extend(gt_test.values())  # Add labels from the test set
        all_labels.extend(gt_train.values())  # Add labels from the train set

        # Count occurrences of each class label
        class_counts = Counter(all_labels)

        # Separate the class counts for plotting
        classes = list(class_counts.keys())
        counts = list(class_counts.values())

        # Plot the class distribution as a bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(classes, counts, color='skyblue')
        plt.xlabel('Class')
        plt.ylabel('Frequency')
        plt.title('Class Distribution in Ground Truth')
        plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
        plt.tight_layout()

        # Save the bar chart as an image in the output folder
        class_distribution_path = os.path.join(
            self.output_folder, "class_distribution.png")
        plt.savefig(class_distribution_path)
        plt.close()

        print(f"Class distribution chart saved to {class_distribution_path}")

    def run_eval(self):
        all_logits = []
        all_labels = []

        for batch_index, batch_val in self.trainer.tqdm(
                enumerate(self.trainer.validation_dataloader),
                total=len(self.trainer.validation_dataloader)):

            self.trainer.dataset_type.move_batch_to_device(
                batch_val, self.trainer.device)
            self.model.inference(self.model.get_batch_input(batch_val))
            self.model.set_loss()
            metrics = self.model.get_metrics()

            logits = self.model.pred[0]  # Shape: (batch_size, 100)
            labels = self.model.target   # Shape: (batch_size,)

            # Convert to list for JSON serialization
            batch_logits = logits.cpu().tolist()
            batch_labels = labels.cpu().tolist()

            all_logits.extend(batch_logits)
            all_labels.extend(batch_labels)

        # Save predictions to JSON file
        with open(self.output_file, "w") as f:
            json.dump({"logits": all_logits, "labels": all_labels}, f)

        print(f"Predictions saved to {self.output_file}")

    def compute_metrics(self):
        """ Compute precision, recall, and F1-score per class, sort them, and save to a text file """
        with open(self.output_file, "r") as f:
            data = json.load(f)

        logits = np.array(data["logits"])
        labels = np.array(data["labels"])

        # Convert logits to predicted classes (argmax)
        preds = np.argmax(logits, axis=1)  # Get class with highest probability

        # Compute per-class precision, recall, and F1-score
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average=None, labels=np.arange(100))

        # Sort classes by F1-score (descending order)
        sorted_indices = np.argsort(-f1)

        # Write PR values to a text file
        with open(self.pr_values_file, "w") as f:
            f.write(
                "Class-wise Precision, Recall, and F1-score (sorted by F1-score):\n\n")
            for i in sorted_indices:
                line = f"Class {i}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1[i]:.4f}\n"
                print(line.strip())  # Print to console as well
                f.write(line)

        print(f"PR values saved to {self.pr_values_file}")

        return precision, recall, f1, sorted_indices

    def plot_pr_curves(self):
        """ Save PR curves for each class and one combined PR curve """
        with open(self.output_file, "r") as f:
            data = json.load(f)

        logits = np.array(data["logits"])
        labels = np.array(data["labels"])

        # Initialize figure for combined PR curve
        plt.figure(figsize=(8, 6))

        for class_id in range(100):
            # Get ground-truth binary labels for this class
            binary_labels = (labels == class_id).astype(int)

            # Get class probabilities from logits
            probs = logits[:, class_id]  # Confidence score for this class

            precision, recall, _ = precision_recall_curve(binary_labels, probs)

            # Save individual class PR curve
            plt.figure(figsize=(6, 4))
            plt.plot(recall, precision, label=f'Class {class_id}', color='b')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve for Class {class_id}')
            plt.legend()
            plt.grid()

            # Save the figure
            file_path = os.path.join(
                self.output_folder, f'pr_curve_class_{class_id}.png')
            plt.savefig(file_path)
            plt.close()

            # Add this class to the combined PR curve
            plt.plot(recall, precision, label=f'Class {class_id}', alpha=0.3)

        # Finalize the combined PR curve
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve for All Classes')
        # Adjust legend for readability
        # plt.legend(loc='upper right', fontsize=6, ncol=5)
        plt.grid()

        # Save the combined PR curve
        combined_pr_path = os.path.join(
            self.output_folder, "pr_curve_all_classes.png")
        plt.savefig(combined_pr_path)
        plt.close()

        print(f"PR curves saved in folder: {self.output_folder}")

    def compute_accuracy(self):
        """
        Compute accuracy from the stored predictions file.
        """
        # Load the predictions file
        try:
            with open(self.output_file, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            print(
                f"Error: {self.output_file} not found. Run `run_eval()` first.")
            return

        # Extract logits and ground truth labels
        logits = np.array(data["logits"])  # Shape: (num_samples, num_classes)
        labels = np.array(data["labels"])  # Shape: (num_samples,)

        # Convert logits to predicted classes (argmax across class probabilities)
        preds = np.argmax(logits, axis=1)  # Get class with highest probability

        # Compute accuracy: (correct predictions) / (total predictions)
        correct_predictions = (preds == labels).sum()
        total_predictions = len(labels)
        accuracy = correct_predictions / total_predictions

        # Print and save accuracy
        accuracy_str = f"Accuracy: {accuracy * 100:.2f}%\n"
        print(accuracy_str.strip())

        with open(self.accuracy_file, "w") as f:
            f.write(accuracy_str)

        print(f"Accuracy saved to {self.accuracy_file}")

    def read_bb_data(self):
        bb_data = []
        with open(self.bb_file, "r") as f:
            for line in f:
                values = line.strip().split(",")  # Split by comma
                if len(values) < 7:  # Ensure valid row format
                    continue
                frame_index, obj_id, x, y, w, h, conf = map(
                    float, values[:7])  # Convert to numbers
                bb_data.append(
                    (int(frame_index) - 1, int(obj_id), x, y, w, h, conf))
        return bb_data

    def run_eval_custom(self):
        self.bb_file
        self.frames_folder
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                0.229, 0.224, 0.225])  # ImageNet
            # transforms.Normalize(mean=[0.548, 0.529, 0.539], std=[0.268, 0.280, 0.274]) # Hockey
        ])
        self.transform_og = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),

            # transforms.Normalize(mean=[0.548, 0.529, 0.539], std=[0.268, 0.280, 0.274]) # Hockey
        ])
        # Read bb text file from self.bb_file in the format of frame_index,id,x,y,w,h,conf,-1,-1,-1,-1
        # Example: 1,1,498.21,443.80,58.48,111.69,0.83,-1,-1,-1
        # Read the frame index from the self.frames folder "frame_0.jpg"...
        # BUT NOTE: The frame index in the bb file is +1 so frame index 1 in the bb file -> corresponsds to frame frame_000000.jpg
        # Now do a loop with tqdm that does the following:
        # Reads 72 Bouding boxes per step
        # Gets their crops as a PIL Image
        # applies the transform
        # Runs the self.model.inference function
        # The self.model.inference function returns the following:
        # return self.pred, self.imgs.shape[0] > 0, legible_mask
        # here pred[0] are the logits of the 100 classes
        # legible mask is the index mask of legible crops
        # Save every legible crop as a PIL image indicating its predicated label in the file name
        # the name should also use the id but a uniquness to it so maybe {id}_{unique}_label.jpg
        # crops should be saved in a new director in the outputfolder
        bb_data = pd.read_csv(self.bb_file, header=None)
        bb_data.columns = ["frame_index", "id", "x", "y", "w",
                           "h", "conf", "dummy1", "dummy2", "dummy3"]

        # Adjust frame indices (since bb file is +1 offset)
        bb_data["frame_index"] -= 1

        # Adjust frame indices (since bb file is +1 offset)

        # Group bounding boxes into chunks of 72 for processing
        batch_size = 72
        num_batches = len(bb_data) // batch_size + \
            (1 if len(bb_data) % batch_size > 0 else 0)

        for batch_start in tqdm.tqdm(range(0, len(bb_data), batch_size), desc="Processing Batches"):
            batch = bb_data.iloc[batch_start:batch_start + batch_size]

            crops = []
            crop_ids = []
            frame_images = {}
            og_crops = []

            # Extract crops
            for _, row in batch.iterrows():
                frame_name = f"frame_{int(row.frame_index)}.jpg"
                frame_path = os.path.join(self.frames_folder, frame_name)

                # Load the frame image if not already loaded
                if frame_name not in frame_images:
                    if not os.path.exists(frame_path):
                        print(f"Warning: Frame {frame_path} not found.")
                        continue
                    frame_images[frame_name] = Image.open(
                        frame_path).convert("RGB")

                # Crop the bounding box
                x, y, w, h = row.x, row.y, row.w, row.h
                crop = frame_images[frame_name].crop((x, y, x + w, y + h))

                # Apply transformations
                crop_transformed = self.transform(crop)
                og_crop = self.transform_og(crop)
                crops.append(crop_transformed)
                og_crops.append(og_crop)
                crop_ids.append(row.id)  # Store ID for later use

            # Convert crops to tensor batch
            if len(crops) == 0:
                continue  # Skip if no crops found
            crops_tensor = torch.stack(crops).to(self.trainer.device)

            # Run inference
            with torch.no_grad():
                pred, has_images, legible_mask = self.model.inference_unlabeled(
                    crops_tensor)

            if not has_images:
                continue

            # Get logits and predicted labels
            logits = pred[0]  # Shape: (num_legible, 100)
            pred_labels = torch.argmax(logits, dim=1).cpu().numpy()

            # Extract legible crops
            legible_indices = legible_mask.cpu().numpy().astype(bool)
            # legible_crops = np.array(crops)[legible_indices]
            legible_crops = np.array(og_crops)[legible_indices]
            legible_ids = np.array(crop_ids)[legible_indices]

            # Save legible crops with predicted labels
            for idx, (crop, crop_id, pred_label) in enumerate(zip(legible_crops, legible_ids, pred_labels)):

                unique_id = f"{int(crop_id)}_{batch_start + idx}"
                crop_filename = f"{unique_id}_{pred_label}.jpg"
                crop = torch.from_numpy(crop)
                crop_path = os.path.join(self.crops_folder, crop_filename)
                crop_pil = transforms.ToPILImage()(crop)  # Convert back to PIL image
                crop_pil.save(crop_path)

        print(f"All cropped images saved to {self.crops_folder}")

        pass


# Save everything in ./preds
evaluator = JerseyEval(output_folder="../logs/eval/jerseyId-eval")
# evaluator.run_eval()  # Save predictions
# evaluator.compute_metrics()  # Calculate precision, recall, F1-score (sorted)
# evaluator.plot_pr_curves()  # Save PR curves for all classes + combined curve
# evaluator.compute_class_distribution()
# evaluator.compute_accuracy()
evaluator.run_eval_custom()
