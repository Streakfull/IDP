import os
import json
import re
import matplotlib.pyplot as plt
import numpy as np


class MultiObjectTrackingEvaluation:
    def __init__(self, base_folder, save_folder, method_map=None):
        self.base_folder = base_folder
        self.save_folder = save_folder
        self.method_map = method_map if method_map else {}
        self.data = {}

    def extract_metrics(self, sort_by='recall'):
        """
        Extract metrics from JSON files, grouped by method name.

        :param sort_by: 'recall' or 'threshold'. Determines sorting order for data.
        """
        for method_name in os.listdir(self.base_folder):
            method_path = os.path.join(self.base_folder, method_name)
            if not os.path.isdir(method_path):
                continue

            self.data[method_name] = []

            for file_name in os.listdir(method_path):
                if file_name.endswith('.json'):
                    file_path = os.path.join(method_path, file_name)
                    with open(file_path, 'r') as file:
                        metrics = json.load(file)

                    # Replace precision=0 with 1
                    if metrics['precision'] == 0:
                        metrics['precision'] = 1

                    if 'threshold_all_frames' in file_name:
                        threshold = 1000  # Assign 1000 for 'threshold_all_frames'

                    # Add threshold information
                    else:
                        threshold_match = re.search(
                            r'threshold_([\d\.]+(?=\b))', file_name)

                        if threshold_match:
                            threshold = float(
                                threshold_match.group(1).rstrip('.'))
                        else:
                            threshold = None

                    metrics['threshold'] = threshold

                    self.data[method_name].append(metrics)

            self.data[method_name].sort(key=lambda x: x[sort_by])

    def plot_pr_curves(self):
        """
        Plot Precision-Recall curves for all methods.
        """
        os.makedirs(self.save_folder, exist_ok=True)

        plt.figure(figsize=(10, 6))
        for method_name, metrics_list in self.data.items():
            recalls = [m['recall'] for m in metrics_list]
            precisions = [m['precision'] for m in metrics_list]

            # Extend last recall value to 1
            if recalls[-1] < 1:
                recalls.append(1)
                precisions.append(precisions[-1])

            human_readable_name = self.method_map.get(method_name, method_name)
            plt.plot(recalls, precisions, label=human_readable_name)

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(self.save_folder, 'pr_curves.png'))
        plt.close()

    def plot_metrics(self):
        """
        Plot each metric aggregated across all methods.
        """
        os.makedirs(self.save_folder, exist_ok=True)

        metrics_to_plot = ['GT', 'idsw', 'FP',
                           'FN', 'MOTP', 'fragments', 'MOTA']
        for metric in metrics_to_plot:
            plt.figure(figsize=(10, 6))

            for method_name, metrics_list in self.data.items():
                # Pair the thresholds with their corresponding metrics
                metrics_with_thresholds = [
                    (m['threshold'], m[metric]) for m in metrics_list]

                # Sort by threshold
                metrics_with_thresholds.sort(key=lambda x: x[0])

                # Unzip the sorted values back into separate lists
                thresholds_sorted, y_sorted = zip(*metrics_with_thresholds)

                # Convert thresholds to percentages (10% intervals)
                max_threshold = max(thresholds_sorted)
                thresholds_percent = [
                    (t / max_threshold) * 100 for t in thresholds_sorted]

                human_readable_name = self.method_map.get(
                    method_name, method_name)
                plt.plot(thresholds_percent, y_sorted,
                         label=human_readable_name)

            plt.xlabel('Frame (%)')
            plt.ylabel(metric)
            plt.title(f'{metric} Across Methods Frames (%)')
            plt.legend()
            plt.grid()
            plt.savefig(os.path.join(self.save_folder,
                        f'{metric}_by_threshold_percent.png'))
            plt.close()
