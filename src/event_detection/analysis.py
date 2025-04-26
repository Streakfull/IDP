import os
import json
import copy
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image
import random


class Analysis:
    def __init__(self, pred_file, out_folder, frames_folder, nms_window=2):
        self.pred_file = pred_file
        self.out_folder = out_folder
        self.frames_folder = frames_folder
        self.nms_window = nms_window

    def _load_json(self, file_path):
        """Load JSON file from the given path."""
        with open(file_path, "r") as f:
            return json.load(f)

    def run(self, exclude_events=None):
        """Run the full analysis pipeline."""
        os.makedirs(self.out_folder, exist_ok=True)

        # Get filtered predictions
        pred = self.get_nms_pred()

        # Count event types, excluding specified events
        event_counts = self._count_event_types(pred, exclude_events)
        self.process_frames(pred)
        self.process_global_frames(pred)

        # Create a bar chart and save it
        # self._plot_event_distribution(event_counts, exclude_events)

    def get_nms_pred(self):
        """Load predictions, apply NMS, and save results."""
        predictions = self._load_json(self.pred_file)
        filtered_predictions = self.non_maximum_suppression(
            predictions, self.nms_window)

        output_file = os.path.join(self.out_folder, "nms_pred.json")
        with open(output_file, "w") as f:
            json.dump(filtered_predictions, f, indent=4)

        print(f"Filtered predictions saved to {output_file}")
        return filtered_predictions

    def non_maximum_suppression(self, pred, window):
        """Perform Non-Maximum Suppression (NMS) on event predictions."""
        new_pred = []
        for video_pred in pred:
            events_by_label = defaultdict(list)
            for e in video_pred['events']:
                events_by_label[e['label']].append(e)

            events = []
            for v in events_by_label.values():
                for e1 in v:
                    for e2 in v:
                        if (
                            e1['frame'] != e2['frame']
                            and abs(e1['frame'] - e2['frame']) <= window
                            and e1['score'] < e2['score']
                        ):
                            break  # Higher score event found
                    else:
                        events.append(e1)

            events.sort(key=lambda x: x['frame'])
            new_video_pred = copy.deepcopy(video_pred)
            new_video_pred['events'] = events
            new_video_pred['num_events'] = len(events)
            new_pred.append(new_video_pred)
        return new_pred

    def _count_event_types(self, predictions, exclude_events=None):
        """Count occurrences of each event type, excluding specified events."""
        exclude_events = set(exclude_events) if exclude_events else set()
        event_counts = defaultdict(int)
        for video_pred in predictions:
            for event in video_pred['events']:
                if event['label'] in exclude_events:
                    continue
                event_counts[event['label']] += 1
        return event_counts

    def _plot_event_distribution(self, event_counts, exclude_events=None):
        """Generate and save a bar chart of event counts, optionally excluding events."""
        plt.figure(figsize=(10, 6))
        plt.bar(event_counts.keys(), event_counts.values(), color="skyblue")
        plt.xlabel("Event Type")
        plt.ylabel("Count")
        title = "Event Distribution"
        if exclude_events:
            title += f" (excluding {', '.join(exclude_events)})"
        plt.title(title)
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Create filename with excluded events if applicable
        exclude_str = "_excluding_" + \
            "_".join(exclude_events) if exclude_events else ""
        output_chart = os.path.join(
            self.out_folder, f"event_distribution{exclude_str}.png")

        plt.savefig(output_chart)
        plt.close()
        print(f"Event distribution chart saved to {output_chart}")

    def process_frames(self, predictions):
        events_by_label = defaultdict(list)
        for video in predictions:
            for event in video['events']:
                events_by_label[event['label']].append(
                    (video['video'], event['frame']))

        for label, events in events_by_label.items():
            event_folder = os.path.join(
                self.out_folder, "event_samples", label)
            os.makedirs(event_folder, exist_ok=True)
            self.create_event_image(label, events, event_folder)

    def create_event_image(self, event_label, events, event_folder):
        """Extract 10 random frames per event type and merge them into separate image files in a 5-row grid."""
        for idx, (video_id, frame) in enumerate(events):
            frame_numbers = [f"{frame + i:06d}.jpg" for i in range(-5, 5)]
            frame_numbers = random.sample(frame_numbers, 10)
            frame_paths = [os.path.join(self.frames_folder, f) for f in frame_numbers if os.path.exists(
                os.path.join(self.frames_folder, f))]
            frame_images = [Image.open(f) for f in frame_paths]

            if frame_images:
                num_cols = 2
                num_rows = 5
                width, height = frame_images[0].size
                combined_image = Image.new(
                    "RGB", (num_cols * width, num_rows * height))

                for i, img in enumerate(frame_images):
                    row, col = divmod(i, num_cols)
                    combined_image.paste(img, (col * width, row * height))

                output_file = os.path.join(event_folder, f"samples_{idx}.jpg")
                combined_image.save(output_file)
                print(f"Saved {output_file}")

    def process_global_frames(self, predictions):
        events_by_label = defaultdict(list)
        for video in predictions:
            for event in video['events']:
                events_by_label[event['label']].append(
                    (video['video'], event['frame']))

        for label, events in events_by_label.items():
            event_folder = os.path.join(
                self.out_folder, "event_samples", label)
            os.makedirs(event_folder, exist_ok=True)
            self.create_global_event_image(label, events, event_folder)

    def create_global_event_image(self, event_label, events, event_folder):
        """Extract 10 random frames from all instances of an event type and merge them into a single image in a 5-row grid."""
        selected_events = random.sample(events, min(10, len(events)))
        frame_paths = [os.path.join(self.frames_folder, f"{frame:06d}.jpg") for _, frame in selected_events if os.path.exists(
            os.path.join(self.frames_folder, f"{frame:06d}.jpg"))]
        frame_images = [Image.open(f) for f in frame_paths]

        if frame_images:
            num_cols = 2
            num_rows = max(1, (len(frame_images) + 1) // 2)
            width, height = frame_images[0].size
            combined_image = Image.new(
                "RGB", (num_cols * width, num_rows * height))

            for i, img in enumerate(frame_images):
                row, col = divmod(i, num_cols)
                combined_image.paste(img, (col * width, row * height))

            output_file = os.path.join(event_folder, "global_samples.jpg")
            combined_image.save(output_file)
            print(f"Saved {output_file}")


# Example Usage
# pred_path = "./logs/full_game/pred-challenge.30.json"
# frames_folder = "./logs/full_game/frames/240"
# output = "./logs/full_game/analysis"


pred_path = "../logs/full_game_ga/nms_pred.json"
frames_folder = "../logs/full_game/frames/240"
output = "../logs/full_game_ga"

analysis = Analysis(pred_file=pred_path, out_folder=output,
                    frames_folder=frames_folder, nms_window=100)

# Run without excluding any event
analysis.run()
# analysis.get_nms_pred()

# Run excluding "PASS" and "DRIVE" events
# analysis.run(exclude_events=["PASS", "DRIVE"])
