import time
from collections import Counter, deque

from enemy_troop_detection import TroopExtractor

# Builds the battlefield state from YOLO's tracked troops (TroopDetector.track):
#   - holds troops that briefly disappear (e.g. hidden in a swarm) instead of dropping them
#   - class = majority vote over recent frames, so one misclassified frame doesn't change it
#   - team  = vote over the first frames of a track: red badge seen -> enemy, otherwise user
#   - position = tile on the game's 18x32 grid
# See docs/troop_tracking_plan.md.

# ================= GRID =================
# Battlefield edges measured on a 900x1600 screenshot (tuned with temp_grid.py). Stored as ratios.
REF_WIDTH, REF_HEIGHT = 900, 1600
ARENA_LEFT, ARENA_RIGHT = 70 / REF_WIDTH, 830 / REF_WIDTH
ARENA_TOP, ARENA_BOTTOM = 155 / REF_HEIGHT, 1190 / REF_HEIGHT
GAME_COLUMNS, GAME_ROWS = 18, 32       # col 0 = left, row 0 = top (enemy side)

# Which point of the box decides the tile:
# "bottom" = bottom-centre (roughly where the troop stands), "centre" = middle of the box
TILE_ANCHOR = "bottom"

# Fixed landmarks, in tiles: (col1, row1, col2, row2), inclusive. Given to the model as a legend.
LANDMARKS = {
    "enemy king tower": (7, 1, 10, 4),
    "enemy left princess tower": (2, 5, 4, 7),
    "enemy right princess tower": (13, 5, 15, 7),
    "river": (0, 15, 17, 16),
    "left bridge": (2, 15, 4, 16),
    "right bridge": (13, 15, 15, 16),
    "user left princess tower": (2, 24, 4, 26),
    "user right princess tower": (13, 24, 15, 26),
    "user king tower": (7, 27, 10, 30),
}

# ================= MEMORY =================
HOLD_SECONDS = 1.5        # keep a troop this long after it disappears before forgetting it
CLASS_HISTORY = 15         # frames of class names kept per troop for the majority vote

# ================= TEAM =================
TEAM_VOTE_FRAMES = 10      # badge checks per new troop; after this the team is fixed
ENEMY_VOTE_RATIO = 0.5     # enemy if at least this fraction of the checks found a red badge


def pixel_to_tile(x, y, frame_width, frame_height):
    """Frame pixel -> (col, row) on the game grid. Points just outside the arena snap to the edge tile."""
    left, right = ARENA_LEFT * frame_width, ARENA_RIGHT * frame_width
    top, bottom = ARENA_TOP * frame_height, ARENA_BOTTOM * frame_height
    col = int((x - left) / (right - left) * GAME_COLUMNS)
    row = int((y - top) / (bottom - top) * GAME_ROWS)
    return (min(max(col, 0), GAME_COLUMNS - 1), min(max(row, 0), GAME_ROWS - 1))


def anchor_point(box):
    x1, y1, x2, y2 = box
    if TILE_ANCHOR == "centre":
        return (x1 + x2) / 2, (y1 + y2) / 2
    return (x1 + x2) / 2, y2


class BoardState:
    def __init__(self):
        self.badge_checker = TroopExtractor()
        self.troops = {}            # track_id -> dict (see _new_troop)
        self.frame_size = None      # (width, height) of the latest frame, for pixel -> tile
        self.last_update = None

    def _new_troop(self, now):
        return {
            "box": None,
            "first_seen": now,
            "last_seen": now,
            "class_history": deque(maxlen=CLASS_HISTORY),
            "team_votes": [],       # True = red badge found in that frame
        }

    def update(self, tracks, frame, now=None):
        """Call once per frame with the output of TroopDetector.track()."""
        now = time.monotonic() if now is None else now
        frame_height, frame_width = frame.shape[:2]
        self.frame_size = (frame_width, frame_height)
        self.last_update = now

        for track_id, class_name, confidence, box in tracks:
            troop = self.troops.setdefault(track_id, self._new_troop(now))
            troop["box"] = box
            troop["last_seen"] = now
            troop["class_history"].append(class_name)

            # Team vote: only for the first TEAM_VOTE_FRAMES frames, then the team is fixed
            if len(troop["team_votes"]) < TEAM_VOTE_FRAMES:
                found_badge = self.badge_checker.find_badge_for_box(frame, box) is not None
                troop["team_votes"].append(found_badge)

        # Forget troops that have been gone longer than the hold time
        for track_id in [tid for tid, troop in self.troops.items() if now - troop["last_seen"] > HOLD_SECONDS]:
            del self.troops[track_id]

    def snapshot(self):
        """Current battlefield: list of troop dicts, sorted by team then position."""
        if self.frame_size is None:
            return []
        frame_width, frame_height = self.frame_size

        troops = []
        for track_id, troop in self.troops.items():
            votes = troop["team_votes"]
            team = "enemy" if sum(votes) / len(votes) >= ENEMY_VOTE_RATIO else "user"
            troops.append({
                "id": track_id,
                "name": Counter(troop["class_history"]).most_common(1)[0][0],
                "team": team,
                "team_decided": len(votes) >= TEAM_VOTE_FRAMES,
                "tile": pixel_to_tile(*anchor_point(troop["box"]), frame_width, frame_height),
                "box": troop["box"],
                "visible": troop["last_seen"] == self.last_update,   # False = being held while hidden
                "age": self.last_update - troop["first_seen"],
            })
        return sorted(troops, key=lambda t: (t["team"], t["tile"][1], t["tile"][0]))

    def describe(self):
        """The battlefield as text lines for the model, e.g. 'enemy giant at (5, 12)'."""
        return [f"{troop['team']} {troop['name']} at {troop['tile']}" for troop in self.snapshot()]


def legend():
    """Fixed description of the grid, sent to the model together with the troop list."""
    lines = [
        f"Grid: {GAME_COLUMNS} columns x {GAME_ROWS} rows, positions are (column, row).",
        f"Column 0 is the left edge, column {GAME_COLUMNS - 1} the right edge.",
        f"Row 0 is the top (enemy side), row {GAME_ROWS - 1} the bottom (user side).",
    ]
    for name, (col1, row1, col2, row2) in LANDMARKS.items():
        lines.append(f"{name}: columns {col1}-{col2}, rows {row1}-{row2}")
    return lines
