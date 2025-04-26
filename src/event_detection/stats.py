import os
import json
import copy
from charset_normalizer import detect
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image
import random


class Stats:
    def __init__(self, pred_file, out_folder, frames_folder, detection_folder, teams_folder, ball_labels_folder):
        self.pred_file = pred_file
        self.out_folder = out_folder
        self.frames_folder = frames_folder
        self.detection_folder = detection_folder
        self.teams_folder = teams_folder
        self.balls_labels_folder = ball_labels_folder
        self.possession = defaultdict(int)
        self.passes = defaultdict(int)
        self.successful_passes = defaultdict(int)
        self.total_passes_with_drive = defaultdict(int)
        self.none_passes = 0
        self.event_counts = defaultdict(lambda: defaultdict(int))

    def _load_json(self, file_path):
        """Load JSON file from the given path."""
        with open(file_path, "r") as f:
            return json.load(f)

    def run(self, exclude_events=None):
        """Run the full analysis pipeline."""
        os.makedirs(self.out_folder, exist_ok=True)
        predictions = self._load_json(self.pred_file)
        self.process_frames(predictions)
        self.calculate_possession()
        self.calculate_possession_time_based()
        self.calculate_pass_accuracy()
        self.calculate_event_counts()

    def process_frames(self, predictions):
        events_by_label = defaultdict(list)
        for video in predictions:
            for event in video['events']:
                if event['label'] == "DRIVE":
                    self.calculate_possession_from_event(event)
                elif event['label'] in ["PASS"]:
                    self.calculate_pass_accuracy_from_event(event, predictions)
                elif event['label'] in ["FREE KICK", "SHOT", "HEADER", "BALL PLAYER BLOCK", "CROSS", "THROW IN"]:
                    self.count_event_per_team(event)

    def calculate_possession_from_event(self, event):
        frame_id = f"{event['frame']:06d}"

        player_id, team_id = self.get_player_with_ball(frame_id)
        if team_id is not None:
            self.possession[team_id] += 1

    def calculate_possession(self):
        total_frames = sum(self.possession.values())
        if total_frames == 0:
            return

        possession_percentage = {
            team: (count / total_frames) * 100 for team, count in self.possession.items()}
        print("Possession Percentage:", possession_percentage)

    def get_player_with_ball2(self, frame_id):
        ball_file = os.path.join(self.balls_labels_folder, f"{frame_id}.txt")

        if not os.path.exists(ball_file):
            return None, None

        with open(ball_file, 'r') as f:
            ball_data = [list(map(float, line.strip().split()))
                         for line in f.readlines()]
        # import pdb
        # pdb.set_trace()
        if not ball_data:
            return None, None

        x_ball, y_ball = (ball_data[0][0] + ball_data[0][2]) / \
            2, (ball_data[0][1] + ball_data[0][3]) / 2

        detection_file = os.path.join(
            self.detection_folder, f"frame_{frame_id}.txt")
        if not os.path.exists(detection_file):
            return None, None

        with open(detection_file, 'r') as f:
            players = [list(map(float, line.strip().split()))
                       for line in f.readlines()]

        if not players:
            return None, None

        min_dist = float('inf')
        closest_player_id = None

        for player in players:
            x_player, y_player = (
                player[0] + player[2]) / 2, (player[1] + player[3]) / 2
            dist = (x_ball - x_player) ** 2 + (y_ball - y_player) ** 2
            if dist < min_dist:
                min_dist = dist
                closest_player_id = int(player[6])

        if closest_player_id is None:
            return None, None

        team_file = os.path.join(self.teams_folder, f"{frame_id}.txt")
        if not os.path.exists(team_file):
            return None, None

        with open(team_file, 'r') as f:
            teams = [list(map(float, line.strip().split()))
                     for line in f.readlines()]

        for team in teams:
            # import pdb
            # pdb.set_trace()
            if int(team[5]) == closest_player_id:
                return closest_player_id, int(team[4])

        return None, None

    def get_player_with_ball(self, frame_id, detection_range=1):
        def find_valid_ball_frame(frame_id):
            frame_int = int(frame_id)
            for offset in range(detection_range+1):
                # Start with current frame, then look before and after
                for direction in [0, -1, 1]:
                    check_frame = f"{frame_int + direction * offset:06d}"
                    ball_file = os.path.join(
                        self.balls_labels_folder, f"{check_frame}.txt")
                    if os.path.exists(ball_file):
                        with open(ball_file, 'r') as f:
                            ball_data = [list(map(float, line.strip().split()))
                                         for line in f.readlines()]
                        if ball_data:
                            return check_frame, ball_data
            return None, None

        valid_frame, ball_data = find_valid_ball_frame(frame_id)
        if not valid_frame:
            return None, None

        x_ball, y_ball = (ball_data[0][0] + ball_data[0][2]) / \
            2, (ball_data[0][1] + ball_data[0][3]) / 2

        detection_file = os.path.join(
            self.detection_folder, f"frame_{valid_frame}.txt")
        if not os.path.exists(detection_file):
            return None, None

        with open(detection_file, 'r') as f:
            players = [list(map(float, line.strip().split()))
                       for line in f.readlines()]

        if not players:
            return None, None

        min_dist = float('inf')
        closest_player_id = None

        for player in players:
            x_player, y_player = (
                player[0] + player[2]) / 2, (player[1] + player[3]) / 2
            dist = (x_ball - x_player) ** 2 + (y_ball - y_player) ** 2
            if dist < min_dist:
                min_dist = dist
                closest_player_id = int(player[6])

        if closest_player_id is None:
            return None, None

        team_file = os.path.join(self.teams_folder, f"{valid_frame}.txt")
        if not os.path.exists(team_file):
            return closest_player_id, None

        with open(team_file, 'r') as f:
            teams = [list(map(float, line.strip().split()))
                     for line in f.readlines()]

        for team in teams:
            if int(team[5]) == closest_player_id:
                return closest_player_id, int(team[4])

        return None, None

    def calculate_possession_time_based(self):
        possession_time = defaultdict(int)
        frame_files = sorted(os.listdir(self.balls_labels_folder))

        for frame_file in frame_files:
            frame_id = frame_file.split('.')[0]
            _, team_id = self.get_player_with_ball(frame_id)
            if team_id is not None:
                possession_time[team_id] += 1

        total_frames = sum(possession_time.values())
        if total_frames == 0:
            return

        possession_percentage = {
            team: (count / total_frames) * 100 for team, count in possession_time.items()}
        print("Possession Percentage (Time-Based):", possession_percentage)

    def calculate_possession(self):
        total_frames = sum(self.possession.values())
        # print(total_frames, "TOTAL")
        if total_frames == 0:
            return

        possession_percentage = {
            team: (count / total_frames) * 100 for team, count in self.possession.items()}
        print("Possession Percentage (Map):", self.possession)
        print("Possession Percentage (Event-Based):", possession_percentage)

    def calculate_pass_accuracy_from_event(self, event, predictions):
        frame_id = event['frame']
        player_id, team_id = self.get_player_with_ball(
            f"{frame_id:06d}", detection_range=10)
        if team_id is None:
            self.none_passes += 1
            return
        self.passes[team_id] += 1

        # for check_frame in range(frame_id + 10, frame_id + 91):
        #     next_player_id, next_team_id = self.get_player_with_ball(
        #         f"{check_frame:06d}")
        #     if next_team_id is not None:
        #         self.total_passes_with_drive[team_id] += 1
        #         if next_team_id == team_id:
        #             self.successful_passes[team_id] += 1
        #         return

        for next_event in predictions[0]['events']:
            if next_event['label'] == "DRIVE" and frame_id < next_event['frame'] <= frame_id + 100:
                next_player_id, next_team_id = self.get_player_with_ball(
                    f"{next_event['frame']:06d}", detection_range=10)
                self.total_passes_with_drive[team_id] += 1
                if next_team_id == team_id:
                    self.successful_passes[team_id] += 1
                break

    def calculate_pass_accuracy(self):
        pass_accuracy = {team: (self.successful_passes[team] / self.passes[team])
                         * 100 if self.passes[team] > 0 else 0 for team in self.passes}
        print("Pass Accuracy:", pass_accuracy)
        print("Total Passes per Team:", self.passes)
        print("Total Passes with Matched Drive per Team:",
              self.total_passes_with_drive)
        print("None passes", self.none_passes)

    def count_event_per_team(self, event):
        frame_id = f"{event['frame']:06d}"
        print(event['label'], event['frame'])
        player_id, team_id = self.get_player_with_ball(
            frame_id, detection_range=10)
        if team_id is not None:
            self.event_counts[event['label']][team_id] += 1

    def calculate_event_counts(self):
        print("Event Counts per Team:", self.event_counts)


pred_file = "../logs/full_game/analysis/nms_pred.json"
out_folder = "../logs/eval/week18/stats"
frames_folder = "../logs/full_game/hd/frames/240"
ball_folder = "../logs/eval/week18/ball-team-supervised_labels/ball"
teams_folder = "../logs/eval/week18/ball-team-supervised_labels/team"
detections_folder = "../logs/eval/week18/labels"


stats = Stats(pred_file=pred_file, out_folder=out_folder,
              frames_folder=frames_folder, detection_folder=detections_folder, teams_folder=teams_folder, ball_labels_folder=ball_folder)


stats.run()
