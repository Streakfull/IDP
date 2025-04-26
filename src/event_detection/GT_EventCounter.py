import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict


class EventCounter:
    def __init__(self, json_file, output_folder):
        self.json_file = json_file
        self.output_folder = output_folder
        self.event_counts = defaultdict(int)
        os.makedirs(self.output_folder, exist_ok=True)

    def load_json(self):
        """Load JSON file."""
        with open(self.json_file, "r") as f:
            return json.load(f)

    def count_events(self):
        """Count occurrences of each event label."""
        data = self.load_json()
        for video in data:
            for event in video['events']:
                self.event_counts[event['label']] += 1

    def plot_event_distribution(self):
        """Plot and save a bar chart of event occurrences."""
        plt.figure(figsize=(12, 6))
        plt.bar(self.event_counts.keys(),
                self.event_counts.values(), color='skyblue')
        plt.xlabel("Event Type")
        plt.ylabel("Count")
        plt.title("Event Distribution")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, "event_distribution.png"))
        plt.close()

    def run(self):
        """Execute the event counting and plotting process."""
        self.count_events()
        self.plot_event_distribution()
        print("Event count chart saved to:", os.path.join(
            self.output_folder, "event_distribution.png"))


# Example usage
json_file = "../logs/soccernet/full-actions-train.json"
output_folder = "../logs/eval/week18/analysis"
event_counter = EventCounter(json_file, output_folder)
event_counter.run()
