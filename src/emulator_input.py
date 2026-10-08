import subprocess

from static_elements_extractor import REGIONS
from board_state import ARENA_LEFT, ARENA_RIGHT, ARENA_TOP, ARENA_BOTTOM, GAME_COLUMNS, GAME_ROWS

# Sends inputs to LDPlayer through ADB. Positions use the same grid as board_state.py
# and the same card regions as static_elements_extractor.py, so what the vision code
# reports (e.g. "enemy giant at (5, 12)") can be fed straight back in as a target tile.

# ================= CONFIG =================
ADB_PATH = r"C:\LDPlayer\LDPlayer14\adb.exe"   # LDPlayer's own adb
DEVICE = "emulator-5554"                        # from `adb devices`
SCREEN_WIDTH, SCREEN_HEIGHT = 900, 1600         # LDPlayer's resolution in portrait (adb reports 1600x900)
CARD_SLOTS = 4


class EmulatorInput:
    def __init__(self, adb_path=ADB_PATH, device=DEVICE):
        self.adb_path = adb_path
        self.device = device

        # Fail early with a clear message if the emulator isn't reachable
        state = subprocess.run([self.adb_path, "-s", self.device, "get-state"],
                               capture_output=True, text=True)
        if state.stdout.strip() != "device":
            raise RuntimeError(f"ADB can't reach {self.device}: {state.stderr.strip() or state.stdout.strip()}"
                               " -- is LDPlayer running?")

    # ================= FOR THE ML =================
    def play_card(self, slot, col, row):
        """Play the card in `slot` (1-4, left to right) on tile (col, row).

        Tile (0, 0) is the top-left (enemy side), (17, 31) the bottom-right (user side).
        The game ignores taps on tiles where the card can't be placed, so no row check is done here
        (spells can target anywhere, and the deploy zone grows when an enemy tower falls).
        """
        self._check_slot(slot)
        self._check_tile(col, row)
        card_x, card_y = self.card_slot_position(slot)
        tile_x, tile_y = self.tile_position(col, row)
        # Both taps in ONE adb command, so it's a single round trip
        self._shell(f"input tap {card_x} {card_y}; input tap {tile_x} {tile_y}")

    def select_card(self, slot):
        """Tap a card slot only (selects the card without placing it)."""
        self._check_slot(slot)
        self.tap(*self.card_slot_position(slot))

    def tap_tile(self, col, row):
        """Tap the centre of a tile only (places the already-selected card)."""
        self._check_tile(col, row)
        self.tap(*self.tile_position(col, row))

    # ================= POSITIONS =================
    @staticmethod
    def card_slot_position(slot):
        """Centre of a card slot in screen pixels."""
        x1, y1, x2, y2 = REGIONS[f"hand_{slot}"]
        return int((x1 + x2) / 2 * SCREEN_WIDTH), int((y1 + y2) / 2 * SCREEN_HEIGHT)

    @staticmethod
    def tile_position(col, row):
        """Centre of a grid tile in screen pixels (same grid as board_state.py)."""
        left, right = ARENA_LEFT * SCREEN_WIDTH, ARENA_RIGHT * SCREEN_WIDTH
        top, bottom = ARENA_TOP * SCREEN_HEIGHT, ARENA_BOTTOM * SCREEN_HEIGHT
        x = left + (col + 0.5) * (right - left) / GAME_COLUMNS
        y = top + (row + 0.5) * (bottom - top) / GAME_ROWS
        return int(x), int(y)

    # ================= LOW LEVEL =================
    def tap(self, x, y):
        """Tap a screen pixel (900x1600 portrait coordinates)."""
        self._shell(f"input tap {int(x)} {int(y)}")

    def _shell(self, command):
        subprocess.run([self.adb_path, "-s", self.device, "shell", command], check=True)

    @staticmethod
    def _check_slot(slot):
        if not 1 <= slot <= CARD_SLOTS:
            raise ValueError(f"card slot must be 1-{CARD_SLOTS}, got {slot}")

    @staticmethod
    def _check_tile(col, row):
        if not (0 <= col < GAME_COLUMNS and 0 <= row < GAME_ROWS):
            raise ValueError(f"tile must be within (0-{GAME_COLUMNS - 1}, 0-{GAME_ROWS - 1}), got ({col}, {row})")
