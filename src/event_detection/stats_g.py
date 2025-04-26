import os
import json
import copy
from charset_normalizer import detect
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image
import random


class Stats:
    def __init__(self, pred_file, out_folder, frames_folder, detection_file):
        self.detection_file = detection_file
        self.pred_file = pred_file
        self.out_folder = out_folder
        self.frames_folder = frames_folder

        self.possession = defaultdict(int)
        self.player_team_map = defaultdict(lambda: "")
        self.possession_player = defaultdict(int)
        self.passes = defaultdict(int)
        self.successful_passes = defaultdict(int)
        self.total_passes_with_drive = defaultdict(int)
        self.none_passes = 0
        self.event_counts = defaultdict(lambda: defaultdict(int))
        self.pass_attempts_player = defaultdict(int)
        self.pass_success_player = defaultdict(int)
        self.event_counts_p = defaultdict()

        self.detections = self._load_detections()

    def _load_json(self, file_path):
        """Load JSON file from the given path."""
        with open(file_path, "r") as f:
            return json.load(f)

    def _load_detections(self):
        """Load detection data from the detection file."""
        detections = []
        with open(self.detection_file, "r") as f:
            for line in f:
                parts = line.strip().split(',')
                frame_id = int(parts[0])
                player_id = parts[1]
                x, y, w, h = map(float, parts[2:6])
                conf = float(parts[6])
                team_id = int(parts[-1])
                detections.append({
                    "frame_id": frame_id,
                    "player_id": player_id,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "conf": conf,
                    "team_id": team_id
                })
        return detections

    def run(self, exclude_events=None):
        """Run the full analysis pipeline."""
        os.makedirs(self.out_folder, exist_ok=True)
        predictions = self._load_json(self.pred_file)
        self.process_frames(predictions)
        self.calculate_possession()
       # self.calculate_possession_time_based()
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
            if player_id is not None:
                self.possession_player[player_id] += 1
                self.player_team_map[player_id] = team_id

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
        frame_id = int(frame_id)
        all_frames = [frame_id + offset * direction
                      for offset in range(detection_range + 1)
                      for direction in [0, -1, 1]]

        for check_frame in all_frames:
            # Get all detections for this frame
            frame_detections = [
                d for d in self.detections if d["frame_id"] == check_frame]
            ball_detections = [
                d for d in frame_detections if d["player_id"] == "B"]
            player_detections = [
                d for d in frame_detections if d["player_id"] != "B"]

            if not ball_detections or not player_detections:
                continue

            ball = ball_detections[0]  # Assume one ball
            x_ball = ball["x"] + ball["w"] / 2
            y_ball = ball["y"] + ball["h"] / 2

            min_dist = float("inf")
            closest_player = None

            for player in player_detections:
                x_player = player["x"] + player["w"] / 2
                y_player = player["y"] + player["h"] / 2
                dist = (x_ball - x_player) ** 2 + (y_ball - y_player) ** 2
                if dist < min_dist:
                    min_dist = dist
                    closest_player = player

            if closest_player:
                return int(closest_player["player_id"]), closest_player["team_id"]

        return None, None

    def calculate_possession_time_based(self):
        possession_time = defaultdict(int)
        frame_files = sorted([f for f in os.listdir(
            self.frames_folder) if f != "fps.txt"])

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
        #########################################################################
        ### PLAYER####
        # Player-level possession relative to their team's total possession from self.possession
        possession_player_percentage = {}
        for player_id, count in self.possession_player.items():
            team_id = self.player_team_map.get(player_id)
            team_total = self.possession.get(team_id, 0)
            if team_total > 0:
                possession_player_percentage[player_id] = (
                    count / team_total) * 100

        # Get top 5 players per team by filtering on possession_player_percentage
        top_players_by_team = defaultdict(list)
        for player_id, percent in possession_player_percentage.items():
            team_id = self.player_team_map.get(player_id)
            top_players_by_team[team_id].append((player_id, percent))

        for team_id in top_players_by_team:
            top_players_by_team[team_id] = sorted(
                top_players_by_team[team_id], key=lambda x: x[1], reverse=True
            )[:5]

        # Debug output
        print("Team Possession Percentage:", possession_percentage)
        print("\nTop 5 Players by Possession per Team (relative to team):")
        for team_id, players in top_players_by_team.items():
            print(f"Team {team_id}:")
            for player_id, percent in players:
                print(f"  Player {player_id}: {percent:.2f}%")

    def calculate_pass_accuracy_from_event(self, event, predictions):
        frame_id = event['frame']
        player_id, team_id = self.get_player_with_ball(
            f"{frame_id:06d}", detection_range=10)
        if team_id is None:
            self.none_passes += 1
            return
        self.passes[team_id] += 1

        if player_id is not None:
            self.pass_attempts_player[player_id] += 1
            self.player_team_map[player_id] = team_id

        for next_event in predictions[0]['events']:
            if next_event['label'] == "DRIVE" and frame_id < next_event['frame'] <= frame_id + 100:
                next_player_id, next_team_id = self.get_player_with_ball(
                    f"{next_event['frame']:06d}", detection_range=10)
                self.total_passes_with_drive[team_id] += 1
                if next_team_id == team_id:
                    self.successful_passes[team_id] += 1
                    if player_id is not None:
                        self.pass_success_player[player_id] += 1
                break

    def calculate_pass_accuracy(self):
        pass_accuracy = {team: (self.successful_passes[team] / self.passes[team])
                         * 100 if self.passes[team] > 0 else 0 for team in self.passes}
        print("Pass Accuracy:", pass_accuracy)
        print("Total Passes per Team:", self.passes)
        print("Total Passes with Matched Drive per Team:",
              self.total_passes_with_drive)
        print("None passes", self.none_passes)
        # Now calculate player-level pass accuracy and number of passes
        pass_accuracy_per_player = {}

        for player_id, pass_attempts in self.pass_attempts_player.items():
            if pass_attempts >= 10:  # Only include players with at least 10 passes
                pass_successes = self.pass_success_player.get(player_id, 0)
                pass_accuracy_per_player[player_id] = (
                    pass_successes / pass_attempts) * 100

        # Group players by team
        team_player_stats = defaultdict(list)
        for player_id, pass_accuracy in pass_accuracy_per_player.items():
            team_id = self.player_team_map.get(player_id)
            if team_id is not None:
                team_player_stats[team_id].append((player_id, pass_accuracy))

        # Get top 5 and worst 5 players per team based on pass accuracy
        top_players_by_team = {}
        worst_players_by_team = {}
        for team_id, players in team_player_stats.items():
            sorted_players = sorted(players, key=lambda x: x[1], reverse=True)

            # Get the top 5 players
            top_players_by_team[team_id] = sorted_players[:5]

            # Get the worst 5 players
            worst_players_by_team[team_id] = sorted_players[-5:]

        # Debug output
        print("Pass Accuracy per Player:", pass_accuracy_per_player)
        print("\nTop 5 Players by Pass Accuracy per Team:")
        for team_id, players in top_players_by_team.items():
            print(f"Team {team_id}:")
            for player_id, accuracy in players:
                pass_attempts = self.pass_attempts_player.get(player_id, 0)
                print(
                    f"  Player {player_id}: {accuracy:.2f}% (Passes: {pass_attempts})")

        print("\nWorst 5 Players by Pass Accuracy per Team:")
        for team_id, players in worst_players_by_team.items():
            print(f"Team {team_id}:")
            for player_id, accuracy in players:
                pass_attempts = self.pass_attempts_player.get(player_id, 0)
                print(
                    f"  Player {player_id}: {accuracy:.2f}% (Passes: {pass_attempts})")

    def count_event_per_team(self, event):

        frame_id = f"{event['frame']:06d}"
        # print(event['label'], event['frame'])
        player_id, team_id = self.get_player_with_ball(
            frame_id, detection_range=10)
        if team_id is not None:
            self.event_counts[event['label']][team_id] += 1
            if player_id is not None:
                if event['label'] not in self.event_counts_p:
                    self.event_counts_p[event['label']] = {
                        "players": defaultdict(lambda: defaultdict(int))}

            # Safely increment the event count for the team and player
            self.event_counts_p[event['label']
                                ]["players"][team_id][player_id] += 1

            # Increment event count per player (for player-level tracking)
            self.event_counts[event['label']][player_id] += 1

            # Indicate which team the player belongs to
            self.player_team_map[player_id] = team_id

    def calculate_event_counts(self):
        print("Event Counts per Team:", self.event_counts)
        import pdb
        pdb.set_trace()


pred_file = "../logs/full_game/analysis/nms_pred.json"
out_folder = "../logs/eval/finalPipeline/stats"
frames_folder = "../logs/full_game/hd/frames/240"
detection_file = "../logs/trackingPipeline-hmreid/liverpoolvsMancityFullGame2Fps/det_files/bb_teams.txt"


stats = Stats(pred_file=pred_file, out_folder=out_folder,
              frames_folder=frames_folder,
              detection_file=detection_file)


stats.run()
