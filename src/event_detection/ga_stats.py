import json
from collections import defaultdict


class EventCounter:
    def __init__(self, pred_file, detection_file):
        self.pred_file = pred_file
        self.detection_file = detection_file
        self.events_per_team = defaultdict(lambda: defaultdict(int))
        self.events_per_player = defaultdict(lambda: defaultdict(int))
        self.detections = self._load_detections()

    def _load_json(self, file_path):
        with open(file_path, "r") as f:
            return json.load(f)

    def _load_detections(self):
        detections = defaultdict(list)
        with open(self.detection_file, "r") as f:
            for line in f:
                parts = line.strip().split(',')
                frame_id = int(parts[0])
                player_id = parts[1]
                x, y, w, h = map(float, parts[2:6])
                conf = float(parts[6])
                team_id = int(parts[-1])
                detections[frame_id].append({
                    "player_id": player_id,
                    "team_id": team_id,
                    "x": x, "y": y, "w": w, "h": h
                })
        return detections

    def get_closest_player(self, frame_id):
        detections = self.detections.get(frame_id, [])
        ball = [d for d in detections if d['player_id'] == 'B']
        players = [d for d in detections if d['player_id'] != 'B']

        if not ball or not players:
            return None, None

        ball = ball[0]
        bx, by = ball['x'] + ball['w'] / 2, ball['y'] + ball['h'] / 2
        closest = min(players, key=lambda p: (
            p['x'] + p['w'] / 2 - bx) ** 2 + (p['y'] + p['h'] / 2 - by) ** 2)
        return int(closest["player_id"]), int(closest["team_id"])

    def count_events(self):
        data = self._load_json(self.pred_file)
        for video_data in data:
            for event in video_data["events"]:
                frame = event["frame"]
                label = event["label"]
                player_id, team_id = self.get_closest_player(frame)

                if team_id is not None:
                    self.events_per_team[label][team_id] += 1
                if player_id is not None:
                    self.events_per_player[label][player_id] += 1

    def print_summary(self):
        print("\nEvents per Team:")
        for label, team_counts in self.events_per_team.items():
            print(f"  {label}:")
            for team_id, count in team_counts.items():
                print(f"    Team {team_id}: {count}")

        print("\nEvents per Player:")
        for label, player_counts in self.events_per_player.items():
            print(f"  {label}:")
            for player_id, count in player_counts.items():
                print(f"    Player {player_id}: {count}")


# Example usage
pred_file = "../logs/full_game_ga/nms_pred.json"
detection_file = "../logs/trackingPipeline-hmreid/liverpoolvsMancityFullGame2Fps/det_files/bb_teams.txt"

counter = EventCounter(pred_file, detection_file)
counter.count_events()
counter.print_summary()
