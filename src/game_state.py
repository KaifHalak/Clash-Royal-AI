import time

from board_state import BoardState, legend
from card_matching import CardMatcher
from elixir_detection_ocr import ElixirExtractor
from match_tracker import MatchTracker
from static_elements_extractor import extract_static_elements
from timer_extractor import TimeExtractor
from tower_tracker import TowerTracker

# Everything the decision maker needs, in one place:
#   - troops on the field (BoardState, updated every frame)
#   - hand, next card, elixir, time left, tower HP (read from the UI)
#   - match memory: plays so far, enemy elixir estimate, enemy cards seen (MatchTracker)
# The UI readers are slower than YOLO, so each one runs on its own interval instead of every frame.

# ================= READ INTERVALS (seconds) =================
CARDS_INTERVAL = 0.5      # hand + next card + elixir (~100 ms together); hand changes when a card is played
TIMER_INTERVAL = 1.0      # timer only changes once a second (~25 ms)
TOWER_INTERVAL = 0.3      # ONE tower per read (~55 ms each), round-robin -> every tower every ~1.8 s

# static_elements_extractor region name -> name used in the game state
TOWERS = {
    "tower_enemy_left": "enemy_left",
    "tower_enemy_right": "enemy_right",
    "tower_enemy_king": "enemy_king",
    "tower_user_left": "user_left",
    "tower_user_right": "user_right",
    "tower_user_king": "user_king",
}


class GameState:
    def __init__(self):
        self.board = BoardState()
        self.card_matcher = CardMatcher()
        self.elixir_reader = ElixirExtractor()
        self.timer_reader = TimeExtractor()
        self.tower_tracker = TowerTracker()
        self.match = MatchTracker()

        # Last known values -- kept when a read fails (e.g. greyed-out cards, OCR misses)
        self.hand = [None, None, None, None]   # card names in slots 1-4
        self.next_card = None
        self.elixir = 0
        self.time_left = None                  # seconds

        self._last_cards_read = 0.0
        self._last_timer_read = 0.0
        self._last_tower_read = 0.0
        self._next_tower = 0                   # index into TOWERS for the round-robin

    def update(self, frame, tracks, now=None):
        """Call once per frame with the frame and the output of TroopDetector.track()."""
        now = time.monotonic() if now is None else now
        self.board.update(tracks, frame, now)
        elements = extract_static_elements(frame)

        if now - self._last_cards_read >= CARDS_INTERVAL:
            self._last_cards_read = now
            for slot in range(4):
                card_name, _ = self.card_matcher.identify(elements[f"hand_{slot + 1}"])
                if card_name is not None:          # greyed-out cards sometimes fail to match
                    self.hand[slot] = card_name
            next_card, _ = self.card_matcher.identify(elements["next_card"])
            if next_card is not None:
                self.next_card = next_card
            elixir, _ = self.elixir_reader.read(elements["elixir_bar"])
            if elixir is not None:
                self.elixir = elixir

        if now - self._last_timer_read >= TIMER_INTERVAL:
            self._last_timer_read = now
            timer = self.timer_reader.read(elements["timer"])
            if timer is not None:
                self.time_left = timer["min"] * 60 + timer["sec"]

        if now - self._last_tower_read >= TOWER_INTERVAL:
            self._last_tower_read = now
            region_name = list(TOWERS)[self._next_tower]
            self.tower_tracker.update(region_name, elements[region_name])
            self._next_tower = (self._next_tower + 1) % len(TOWERS)

        self.match.update(self.board.snapshot(), self.time_left, self.elixir, self.towers(), now)

    def towers(self):
        """{"enemy_left": {"hp": 1512, "dead": False}, ...} -- hp None = not read yet / not shown."""
        towers = {}
        for region_name, name in TOWERS.items():
            seen = region_name in self.tower_tracker.towers   # TowerTracker creates entries on first update
            towers[name] = {
                "hp": self.tower_tracker.get_last_known_hp(region_name) if seen else None,
                "dead": self.tower_tracker.is_dead(region_name) if seen else False,
            }
        return towers

    def snapshot(self):
        """The whole game state as one dict.

        Example payload:
        {
            "time_left": 71,                                  # seconds, None until the timer is read
            "elixir": 6,
            "hand": ["giant", "arrows", "mini_pekka", "musketeer"],   # slots 1-4
            "next_card": "knight",
            "towers": {
                "enemy_left":  {"hp": 1512, "dead": False},   # hp None = not read yet / not shown
                "enemy_right": {"hp": 1206, "dead": False},
                "enemy_king":  {"hp": None, "dead": False},
                "user_left":   {"hp": 308,  "dead": False},
                "user_right":  {"hp": 1512, "dead": False},
                "user_king":   {"hp": None, "dead": False},
            },
            "troops": [ ... ],                                # BoardState.snapshot()
            "match": { ... },                                 # MatchTracker.snapshot()
        }
        """
        return {
            "time_left": self.time_left,
            "elixir": self.elixir,
            "hand": list(self.hand),
            "next_card": self.next_card,
            "towers": self.towers(),
            "troops": self.board.snapshot(),
            "match": self.match.snapshot(),
        }

    def describe(self):
        """The game state as text lines for a model (the grid legend is separate: legend()).

        Example payload:
        [
            "time left: 1:11",
            "elixir: 6",
            "hand: slot 1 giant, slot 2 arrows, slot 3 mini_pekka, slot 4 musketeer",
            "next card: knight",
            "towers: enemy_left 1512, enemy_right 1206, enemy_king ?, user_left 308, user_right 1512, user_king ?",
            "phase: single elixir | score: us 0 - 1 them",    # MatchTracker.describe()
            "enemy elixir (est.): 3.6",
            ...
            "troops:",
            "enemy knight at (14, 22)",
            "user giant at (4, 22)",
        ]
        """
        state = self.snapshot()
        time_left = "?" if state["time_left"] is None else f"{state['time_left'] // 60}:{state['time_left'] % 60:02d}"
        hand = ", ".join(f"slot {slot} {card or '?'}" for slot, card in enumerate(state["hand"], start=1))
        towers = ", ".join(
            f"{name} {'DEAD' if tower['dead'] else (tower['hp'] if tower['hp'] is not None else '?')}"
            for name, tower in state["towers"].items()
        )
        return [
            f"time left: {time_left}",
            f"elixir: {state['elixir']}",
            f"hand: {hand}",
            f"next card: {state['next_card'] or '?'}",
            f"towers: {towers}",
            *self.match.describe(),
            "troops:",
            *(self.board.describe() or ["(none)"]),
        ]

    @staticmethod
    def legend():
        return legend()
