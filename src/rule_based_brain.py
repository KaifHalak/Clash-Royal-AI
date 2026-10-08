import time

# Rule-based decision maker: board state + hand + elixir -> which card to play where (or wait).
# Rules are checked in priority order; the first one that fires picks the action.
#   1. Defend   - an enemy troop is on our side / crossing a bridge
#   2. Spell    - an enemy target is worth a Fireball
#   3. Support  - our Giant is walking -> put a ranged card behind it
#   4. Elixir   - elixir is nearly full -> start a push instead of wasting it
#   5. Wait

# ================= CARDS =================
# Our deck, keyed by the card matcher's names (all-cards/*.png): cost + role
CARDS = {
    "giant":      {"cost": 5, "role": "tank"},
    "mini_pekka": {"cost": 4, "role": "melee"},
    "knight":     {"cost": 3, "role": "melee"},
    "musketeer":  {"cost": 4, "role": "ranged"},
    "archers":    {"cost": 3, "role": "ranged"},
    "minions":    {"cost": 3, "role": "air"},
    "arrows":     {"cost": 3, "role": "spell"},
    "fireball":   {"cost": 4, "role": "spell"},
}

# Units on the board, keyed by YOLO's class names: what kind of threat each one is
UNIT_TYPES = {
    "giant": "tank",
    "knight": "melee",
    "mini-pekka": "melee",
    "musketeer": "ranged",
    "archer": "ranged",
    "minion": "swarm",
    "goblin": "swarm",
    "spr-goblin": "swarm",
    "goblin-hut": "building",
    "arrows": "spell",
    "fireball": "spell",
}

# Which of our cards to answer each threat type with, best first
COUNTERS = {
    "tank":   ["mini_pekka", "knight", "minions"],
    "melee":  ["minions", "mini_pekka", "knight", "archers"],
    "ranged": ["knight", "mini_pekka", "minions"],
    "swarm":  ["archers", "musketeer", "knight"],     # arrows handled separately (needs a group)
}

# ================= POSITIONS (tiles, same grid as board_state.py) =================
LANE_COLUMNS = {"left": 3, "right": 14}   # centre column of each lane / princess tower
DEFEND_FROM_ROW = 15      # enemy troop at this row or lower (bridge + our side) = threat
MELEE_DEFEND_ROW = 21     # 2-3 tiles in front of our princess towers (rows 24-26)
RANGED_DEFEND_ROW = 27    # just behind our princess towers
OUR_FIRST_ROW = 17        # first row on our side of the river (deploy zone starts here)
BEHIND_KING_ROW = 31      # bottom row, behind our king tower
BRIDGE_ROW = OUR_FIRST_ROW

# ================= TUNING =================
PLAY_COOLDOWN = 1.0       # seconds to wait after playing a card before playing another
ARROWS_MIN_GROUP = 3      # enemy swarm units needed to spend Arrows
FIREBALL_MIN_GROUP = 2    # enemy ranged units needed for Fireball (a lone Musketeer also counts)
GROUP_RADIUS = 2          # tiles; units within this distance count as one group
FULL_ELIXIR = 9           # at/above this, play something instead of wasting elixir
SUPPORT_DISTANCE = 3      # tiles behind our Giant to place the support card
DEFAULT_PUSH_LANE = "left"


def lane_of(tile):
    return "left" if tile[0] < 9 else "right"


def tile_distance(tile_a, tile_b):
    """Tiles between two positions (Chebyshev: diagonal steps count as 1)."""
    return max(abs(tile_a[0] - tile_b[0]), abs(tile_a[1] - tile_b[1]))


def biggest_group(troops):
    """The troop with the most others within GROUP_RADIUS, as (centre_tile, size)."""
    best_centre, best_size = None, 0
    for troop in troops:
        group = [other for other in troops if tile_distance(troop["tile"], other["tile"]) <= GROUP_RADIUS]
        if len(group) > best_size:
            cols = [other["tile"][0] for other in group]
            rows = [other["tile"][1] for other in group]
            best_centre = (round(sum(cols) / len(cols)), round(sum(rows) / len(rows)))
            best_size = len(group)
    return best_centre, best_size


class RuleBasedBrain:
    def __init__(self, push_lane=DEFAULT_PUSH_LANE):
        self.push_lane = push_lane
        self.last_play_time = None

    def decide(self, troops, hand, elixir, now=None):
        """Pick the next action.

        troops: BoardState.snapshot()
        hand:   card names in slots 1-4, e.g. ["giant", "arrows", "knight", "musketeer"] (None = unknown)
        elixir: current elixir (0-10)

        Returns None (wait) or an action dict:
            {"slot": 2, "card": "arrows", "tile": (4, 18), "reason": "arrows on 3 swarm units"}
        """
        now = time.monotonic() if now is None else now
        if self.last_play_time is not None and now - self.last_play_time < PLAY_COOLDOWN:
            return None

        # Cards we can play right now: {card: slot}
        playable = {card: slot for slot, card in enumerate(hand, start=1)
                    if card in CARDS and CARDS[card]["cost"] <= elixir}
        if not playable:
            return None

        # Only trust troops whose team vote has finished
        enemies = [t for t in troops if t["team"] == "enemy" and t["team_decided"]]
        ours = [t for t in troops if t["team"] == "user" and t["team_decided"]]

        for rule in (self._defend, self._spell, self._support, self._spend_elixir):
            action = rule(enemies, ours, playable, elixir)
            if action is not None:
                self.last_play_time = now
                return action
        return None

    # ================= RULES =================
    def _defend(self, enemies, ours, playable, elixir):
        """Rule 1: answer the enemy troop closest to our towers, once it reaches the bridge."""
        threats = [t for t in enemies
                   if t["tile"][1] >= DEFEND_FROM_ROW and UNIT_TYPES.get(t["name"]) not in (None, "building", "spell")]
        if not threats:
            return None

        # Swarm on our side: Arrows if enough units are grouped together
        swarm = [t for t in threats if UNIT_TYPES[t["name"]] == "swarm"]
        if "arrows" in playable and swarm:
            centre, size = biggest_group(swarm)
            if size >= ARROWS_MIN_GROUP:
                return self._action(playable, "arrows", centre, f"arrows on {size} swarm units")

        threat = max(threats, key=lambda t: t["tile"][1])          # deepest = closest to our towers
        threat_type = UNIT_TYPES[threat["name"]]
        lane = lane_of(threat["tile"])
        for card in COUNTERS[threat_type]:
            if card not in playable:
                continue
            if CARDS[card]["role"] == "ranged":
                tile = (LANE_COLUMNS[lane], RANGED_DEFEND_ROW)
            else:
                # In front of the tower, or right next to the threat if it's already past that point
                tile = (LANE_COLUMNS[lane], max(MELEE_DEFEND_ROW, min(threat["tile"][1] + 1, BEHIND_KING_ROW)))
            return self._action(playable, card, tile, f"defend {threat['name']} at {threat['tile']} with {card}")
        return None

    def _spell(self, enemies, ours, playable, elixir):
        """Rule 2: Fireball a Musketeer or a group of ranged troops, anywhere on the field."""
        if "fireball" not in playable:
            return None
        ranged = [t for t in enemies if UNIT_TYPES.get(t["name"]) == "ranged"]
        if not ranged:
            return None
        centre, size = biggest_group(ranged)
        has_musketeer = any(t["name"] == "musketeer" for t in ranged)
        if size >= FIREBALL_MIN_GROUP or has_musketeer:
            return self._action(playable, "fireball", centre, f"fireball on {size} ranged units")
        return None

    def _support(self, enemies, ours, playable, elixir):
        """Rule 3: our Giant is on the field -> put a ranged card behind it (once)."""
        giants = [t for t in ours if t["name"] == "giant"]
        if not giants:
            return None
        giant = giants[0]
        already_supported = any(UNIT_TYPES.get(t["name"]) == "ranged" and lane_of(t["tile"]) == lane_of(giant["tile"])
                                for t in ours)
        if already_supported:
            return None
        for card in ("musketeer", "archers"):
            if card in playable:
                # Behind the Giant, but never past our side of the river
                tile = (giant["tile"][0], max(giant["tile"][1] + SUPPORT_DISTANCE, OUR_FIRST_ROW))
                return self._action(playable, card, tile, f"support giant at {giant['tile']} with {card}")
        return None

    def _spend_elixir(self, enemies, ours, playable, elixir):
        """Rule 4: elixir nearly full -> Giant behind the king tower, else cheapest troop at the bridge."""
        if elixir < FULL_ELIXIR:
            return None
        lane_col = LANE_COLUMNS[self.push_lane]
        if "giant" in playable:
            king_side_col = 8 if self.push_lane == "left" else 9
            return self._action(playable, "giant", (king_side_col, BEHIND_KING_ROW), "elixir full: giant from the back")
        troops = [card for card in playable if CARDS[card]["role"] != "spell"]
        if troops:
            cheapest = min(troops, key=lambda card: CARDS[card]["cost"])
            return self._action(playable, cheapest, (lane_col, BRIDGE_ROW), f"elixir full: {cheapest} at the bridge")
        return None

    @staticmethod
    def _action(playable, card, tile, reason):
        return {"slot": playable[card], "card": card, "tile": tile, "reason": reason}
