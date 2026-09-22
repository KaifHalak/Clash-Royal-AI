"""
Tracks tower HP over time, per tower, and infers when a tower is dead.
Sits between TowerHealthExtractor (pure OCR: crop -> number) and main --
main only ever talks to this class, never to the extractor directly.

Assumes TowerHealthExtractor.read(crop) is stateless: returns an int or
None, with no death-tracking logic of its own.
"""

import time
from tower_health import TowerHealthExtractor


class TowerTracker:
    def __init__(self):
        self.extractor = TowerHealthExtractor()  # one shared TowerHealthExtractor instance -- one model load

        self.death_grace_period_sec = 3.0            # seconds of None before considering death
        self.death_hp_ratio_threshold = 0.20  # last_known_hp must be below this fraction of max_hp

        # {tower_name: {"max_hp", "last_known_hp", "last_healthy_time", "is_dead"}}
        self.towers = {}

    def _get_tower_state(self, tower_name):
        """Returns this tower's tracking state, creating a fresh entry the first time it's seen."""
        if tower_name not in self.towers:
            self.towers[tower_name] = {
                "max_hp": None,
                "last_known_hp": None,
                "last_healthy_time": None,
                "is_dead": False,
            }
        return self.towers[tower_name]

    def _check_alive(self, tower_state, tower_hp):
        """
        Updates state from the latest reading, returns whether the tower
        should now be considered dead.

        Rule: a None reading only counts toward death once last_known_hp has
        dropped below death_hp_ratio_threshold of max_hp -- this stops a
        temporary OCR miss on a healthy tower (spell effect, bad frame,
        animation) from ever being mistaken for death, since HP can't jump
        from mid/high straight to "about to die".
        
        True: Tower is dead
        False: Tower is alive
        """
        now = time.time()

        if tower_hp is not None:
            if tower_state["max_hp"] is None:
                tower_state["max_hp"] = tower_hp  # set the first successful read as max health

            tower_state["last_known_hp"] = tower_hp
            tower_state["last_healthy_time"] = now
            return False

        if tower_state["last_known_hp"] is None or tower_state["last_healthy_time"] is None:
            return False  # never had a healthy reading yet -- nothing to judge from

        low_hp = tower_state["last_known_hp"] <= tower_state["max_hp"] * self.death_hp_ratio_threshold
        long_enough_gone = (now - tower_state["last_healthy_time"]) >= self.death_grace_period_sec

        return low_hp and long_enough_gone

    def update(self, tower_name, crop):
        """
        Reads and tracks one tower's HP for this poll.
        Returns the current HP (int or None). Once a tower is confirmed
        dead, OCR is skipped entirely for it going forward.
        """
        tower_state = self._get_tower_state(tower_name)

        if tower_state["is_dead"]:
            return None

        tower_hp = self.extractor.read(crop)

        if self._check_alive(tower_state, tower_hp):
            tower_state["is_dead"] = True

        return tower_hp

    def is_dead(self, tower_name):
        # True: Tower is dead
        return self._get_tower_state(tower_name)["is_dead"]

    def is_king_active(self, tower_name):
        # True: King tower is active
        tower_state = self._get_tower_state(tower_name)
        return not tower_state["is_dead"] and tower_state["last_known_hp"] is not None

    def get_last_known_hp(self, tower_name):
        return self._get_tower_state(tower_name)["last_known_hp"]

    def reset(self):
        """Call between matches so the previous game's tower states don't leak in."""
        self.towers = {}