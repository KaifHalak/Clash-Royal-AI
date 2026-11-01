from collections import deque

# Match memory: turns "troops appeared on the board" into "the enemy played card X", and keeps the
# history the brain needs:
#   - every play (ours + the enemy's): card, cost, tile, lane, when
#   - enemy elixir estimate (their plays + regen at the current phase's rate)
#   - enemy cards seen so far + their last plays (a card just played is not in their hand)
#   - game phase (single / double elixir) and score (towers taken)
# It is approximate on purpose: it would rather MISS an enemy play than invent one, and the elixir
# estimate corrects itself by staying within 0-10.

# ================= CARDS =================
# YOLO class name -> the card that spawns it: card name, cost, units one card spawns.
# "units" caps how many same-type units count as ONE card. Too high only merges two quick casts into
# one (a missed play); too low splits one cast into two (an invented play), so round up when unsure.
UNIT_CARDS = {
    "giant":      {"card": "giant",         "cost": 5, "units": 1},
    "knight":     {"card": "knight",        "cost": 3, "units": 1},
    "mini-pekka": {"card": "mini_pekka",    "cost": 4, "units": 1},
    "musketeer":  {"card": "musketeer",     "cost": 4, "units": 1},
    "archer":     {"card": "archers",       "cost": 3, "units": 2},
    "minion":     {"card": "minions",       "cost": 3, "units": 3},
    "goblin":     {"card": "goblins",       "cost": 2, "units": 4},
    "spr-goblin": {"card": "spear_goblins", "cost": 2, "units": 3},
    "goblin-hut": {"card": "goblin_hut",    "cost": 5, "units": 1},
    "arrows":     {"card": "arrows",        "cost": 3, "units": 1},
    "fireball":   {"card": "fireball",      "cost": 4, "units": 1},
}
SPELLS = {"arrows", "fireball"}                      # YOLO names that are spells (no team badge)
CARD_COSTS = {info["card"]: info["cost"] for info in UNIT_CARDS.values()}
DECK_SIZE = 8
HAND_SIZE = 4

# ================= PLAY DETECTION (tiles / seconds) =================
GROUP_SECONDS = 1.5       # same-type units first seen within this time of each other ...
GROUP_RADIUS = 3          # ... and this close together are one card
ENEMY_DEPLOY_MAX_ROW = 15  # enemy troops first seen below this row aren't fresh plays (lost + re-found track)
POCKET_MAX_ROW = 23       # ... unless our princess tower in that lane is down (enemy can deploy deeper)
REFOUND_SECONDS = 3.0     # a new unit near a same-type unit that vanished this recently is the same unit
REFOUND_RADIUS = 3
HUT_SPAWN_RADIUS = 3      # spear goblins this close to an enemy goblin hut were spawned by it
OUR_PLAY_SECONDS = 3.0    # a unit near our own play of the same card this recently is ours,
OUR_PLAY_RADIUS = 3       # whatever the team vote says (catches badge mis-votes)

# ================= ELIXIR / CLOCK =================
MAX_ELIXIR = 10
ELIXIR_SECONDS = 2.8      # seconds per elixir in single elixir
DOUBLE_ELIXIR_TIME = 60   # time_left at/below this = double elixir (overtime is not told apart yet)
NEW_MATCH_JUMP = 30       # time_left going UP by this much = a new match started -> reset

RECENT_PLAYS_SHOWN = 4    # plays listed by describe()


def tile_distance(tile_a, tile_b):
    """Tiles between two positions (Chebyshev: diagonal steps count as 1)."""
    return max(abs(tile_a[0] - tile_b[0]), abs(tile_a[1] - tile_b[1]))


def lane_of(tile):
    return "left" if tile[0] < 9 else "right"


def clock(time_left):
    return "?:??" if time_left is None else f"{time_left // 60}:{time_left % 60:02d}"


class MatchTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.plays = []                 # every play, oldest first (see _add_play)
        self.enemy_elixir = 5.0         # replaced by our own elixir until someone plays (both start equal)
        self.time_left = None
        self.towers = {}
        self.last_update = None
        self.first_sight = {}           # track_id -> (tile, time) where/when the track was first seen
        self.processed = set()          # track ids already classified (counted, or ignored)
        self.previous = {}              # track_id -> troop dict from the last update
        self.lost = deque()             # (name, tile, time) of troops the board just dropped
        self.groups = []                # open casts: {"team", "name", "tile", "time", "units"}

    # ================= INPUT =================
    def update(self, troops, time_left, elixir, towers, now):
        """Call once per frame, after BoardState.update().

        troops:    BoardState.snapshot()
        time_left: seconds (None until the timer is read)
        elixir:    our elixir (used as the enemy's until anyone has played)
        towers:    GameState tower dict, e.g. {"enemy_left": {"hp": 1512, "dead": False}, ...}
        """
        if time_left is not None and self.time_left is not None and time_left > self.time_left + NEW_MATCH_JUMP:
            self.reset()
        self._regen(now, time_left)
        if not self.plays:
            self.enemy_elixir = float(elixir)
        if time_left is not None:
            self.time_left = time_left
        self.towers = towers
        self.last_update = now

        current = {troop["id"]: troop for troop in troops}
        for track_id, troop in current.items():
            if track_id not in self.first_sight:
                self.first_sight[track_id] = (troop["tile"], now - troop["age"])

        # Troops the board dropped since the last update: remember them, a new ID may be the same unit
        for track_id, troop in self.previous.items():
            if track_id not in current:
                self.lost.append((troop["name"], troop["tile"], now))
                self.first_sight.pop(track_id, None)
                self.processed.discard(track_id)
        while self.lost and now - self.lost[0][2] > REFOUND_SECONDS:
            self.lost.popleft()
        self.groups = [group for group in self.groups if now - group["time"] <= GROUP_SECONDS]

        for track_id, troop in current.items():
            if track_id in self.processed or troop["name"] not in UNIT_CARDS:
                continue
            if troop["name"] in SPELLS:
                self.processed.add(track_id)
                self._new_spell(troop, now)
            elif troop["team_decided"]:                  # wait for the badge vote before judging a unit
                self.processed.add(track_id)
                self._new_unit(troop, current, now)

        self.previous = current

    def record_our_play(self, card, tile, now):
        """Call right after the bot plays a card (we know our own plays, no need to detect them)."""
        self._add_play("user", card, CARD_COSTS.get(card, 0), tile, now)

    # ================= PLAY DETECTION =================
    def _new_unit(self, troop, current, now):
        name = troop["name"]
        tile, _ = self.first_sight[troop["id"]]
        if troop["team"] == "user" or self._near_our_play(UNIT_CARDS[name]["card"], tile, now):
            return                                       # ours: logged by record_our_play()
        if name == "spr-goblin" and any(
                other["name"] == "goblin-hut" and other["team"] == "enemy"
                and tile_distance(other["tile"], tile) <= HUT_SPAWN_RADIUS for other in current.values()):
            return                                       # spawned by their goblin hut, not a play
        if self._refound(troop, tile, current, now):
            return
        max_row = POCKET_MAX_ROW if self._our_tower_dead(lane_of(tile)) else ENEMY_DEPLOY_MAX_ROW
        if tile[1] > max_row:
            return                                       # first seen deep on our side: an old unit, not a play
        if self._join_group("enemy", name, tile, now):
            return
        self._add_play("enemy", UNIT_CARDS[name]["card"], UNIT_CARDS[name]["cost"], tile, now)
        self.groups.append({"team": "enemy", "name": name, "tile": tile, "time": now, "units": 1})

    def _new_spell(self, troop, now):
        """Spells carry no badge and vanish before a team vote, so: ours if we just cast it, else theirs."""
        name = troop["name"]
        tile, _ = self.first_sight[troop["id"]]
        if self._join_group(None, name, tile, now):
            return                                       # same cast, seen as several boxes / IDs
        card = UNIT_CARDS[name]["card"]
        ours = any(play["team"] == "user" and play["card"] == card and now - play["time"] <= OUR_PLAY_SECONDS
                   for play in self.plays)
        team = "user" if ours else "enemy"
        if not ours:
            self._add_play("enemy", card, UNIT_CARDS[name]["cost"], tile, now)
        self.groups.append({"team": team, "name": name, "tile": tile, "time": now, "units": 1})

    def _join_group(self, team, name, tile, now):
        """Add the unit to an open cast of the same type nearby, if that cast still has room. team None = any."""
        for group in self.groups:
            if (group["name"] == name and (team is None or group["team"] == team)
                    and tile_distance(group["tile"], tile) <= GROUP_RADIUS
                    and group["units"] < UNIT_CARDS[name]["units"]):
                group["units"] += 1
                group["time"] = now
                return True
        return False

    def _refound(self, troop, tile, current, now):
        """A same-type unit nearby is hidden right now or just vanished -> this is that unit with a new ID."""
        for other in current.values():
            if (other["id"] != troop["id"] and not other["visible"] and other["name"] == troop["name"]
                    and tile_distance(other["tile"], tile) <= REFOUND_RADIUS):
                return True
        for index, (name, lost_tile, _) in enumerate(self.lost):
            if name == troop["name"] and tile_distance(lost_tile, tile) <= REFOUND_RADIUS:
                del self.lost[index]                     # each vanished unit explains one new ID
                return True
        return False

    def _near_our_play(self, card, tile, now):
        return any(play["team"] == "user" and play["card"] == card and now - play["time"] <= OUR_PLAY_SECONDS
                   and tile_distance(play["tile"], tile) <= OUR_PLAY_RADIUS for play in self.plays)

    def _our_tower_dead(self, lane):
        return self.towers.get(f"user_{lane}", {}).get("dead", False)

    def _add_play(self, team, card, cost, tile, now):
        self.plays.append({
            "team": team,
            "card": card,
            "cost": cost,
            "tile": tuple(tile),
            "lane": lane_of(tile),
            "time": now,                    # time.monotonic()
            "time_left": self.time_left,    # match clock when it was played (None if not read yet)
        })
        if team == "enemy":
            self.enemy_elixir = max(0.0, self.enemy_elixir - cost)   # went below 0 -> they had more, clamp

    def _regen(self, now, time_left):
        if self.last_update is None or time_left is None:
            return                          # no timer yet = match not started
        rate = 2 if time_left <= DOUBLE_ELIXIR_TIME else 1
        self.enemy_elixir = min(MAX_ELIXIR, self.enemy_elixir + (now - self.last_update) * rate / ELIXIR_SECONDS)

    # ================= QUERIES =================
    def enemy_plays(self):
        return [play for play in self.plays if play["team"] == "enemy"]

    def enemy_cards_seen(self):
        """Enemy cards in the order they were first seen (their deck, as far as we know it)."""
        return list(dict.fromkeys(play["card"] for play in self.enemy_plays()))

    def enemy_out_of_hand(self):
        """Their last 4 plays, newest first: a played card goes to the back of the queue for 4 plays."""
        return list(dict.fromkeys(play["card"] for play in reversed(self.enemy_plays())))[:HAND_SIZE]

    def enemy_hand_guess(self):
        """Their probable hand once their whole deck has been seen, else None."""
        seen = self.enemy_cards_seen()
        if len(seen) < DECK_SIZE:
            return None
        out = self.enemy_out_of_hand()
        return [card for card in seen if card not in out]

    def phase(self):
        if self.time_left is None:
            return None
        return "double" if self.time_left <= DOUBLE_ELIXIR_TIME else "single"

    def score(self):
        """Crowns: towers each side has destroyed (king tower = 3)."""
        def crowns(side):
            if self.towers.get(f"{side}_king", {}).get("dead"):
                return 3
            return sum(self.towers.get(f"{side}_{lane}", {}).get("dead", False) for lane in ("left", "right"))
        return {"user": crowns("enemy"), "enemy": crowns("user")}

    def snapshot(self):
        """Everything above as one dict.

        Example payload:
        {
            "phase": "single",                       # None until the timer is read
            "score": {"user": 0, "enemy": 1},
            "enemy_elixir": 3.6,                     # estimate
            "enemy_cards_seen": ["giant", "archers", "arrows"],
            "enemy_out_of_hand": ["arrows", "giant", "archers"],
            "enemy_hand_guess": None,                # list once all 8 enemy cards have been seen
            "plays": [ {"team": "enemy", "card": "giant", "cost": 5, "tile": (3, 2), "lane": "left",
                        "time": 1234.5, "time_left": 151}, ... ],
        }
        """
        return {
            "phase": self.phase(),
            "score": self.score(),
            "enemy_elixir": round(self.enemy_elixir, 1),
            "enemy_cards_seen": self.enemy_cards_seen(),
            "enemy_out_of_hand": self.enemy_out_of_hand(),
            "enemy_hand_guess": self.enemy_hand_guess(),
            "plays": list(self.plays),
        }

    def describe(self):
        """The match memory as text lines for the status panel / a model.

        Example payload:
        [
            "phase: single elixir | score: us 0 - 1 them",
            "enemy elixir (est.): 3.6",
            "enemy cards seen (3/8): giant, archers, arrows",
            "enemy not in hand: arrows, giant, archers",
            "plays: 2:31 enemy giant left | 2:20 user knight left | 2:12 enemy archers left",
        ]
        """
        phase = self.phase()
        score = self.score()
        seen = self.enemy_cards_seen()
        hand = self.enemy_hand_guess()
        recent = " | ".join(f"{clock(play['time_left'])} {play['team']} {play['card']} {play['lane']}"
                            for play in reversed(self.plays[-RECENT_PLAYS_SHOWN:]))
        return [
            f"phase: {phase + ' elixir' if phase else '?'} | score: us {score['user']} - {score['enemy']} them",
            f"enemy elixir (est.): {self.enemy_elixir:.1f}",
            f"enemy cards seen ({len(seen)}/{DECK_SIZE}): {', '.join(seen) or 'none'}",
            f"enemy not in hand: {', '.join(self.enemy_out_of_hand()) or '?'}",
            *([f"enemy hand (guess): {', '.join(hand)}"] if hand else []),
            f"plays: {recent or 'none'}",
        ]
