import os
import subprocess
import sys
import time

# Test script: deploy one card through ADB by tapping the card slot, then the target tile.
# Run it during a trainer match with the card in hand:
#   .venv\Scripts\python.exe temp_input.py              -> uses SLOT and TILE below
#   .venv\Scripts\python.exe temp_input.py 2 4 20       -> slot 2 at tile (4, 20)

# Reuse the existing screen regions and grid, so taps use the exact same coordinates as the vision code
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from static_elements_extractor import REGIONS
from board_state import ARENA_LEFT, ARENA_RIGHT, ARENA_TOP, ARENA_BOTTOM, GAME_COLUMNS, GAME_ROWS

# ================= CONFIG =================
ADB = r"C:\LDPlayer\LDPlayer14\adb.exe"
DEVICE = "emulator-5554"
SCREEN_WIDTH, SCREEN_HEIGHT = 900, 1600   # LDPlayer's resolution in portrait (adb reports 1600x900)

SLOT = 1              # card slot 1-4 (left to right)
TILE = (4, 20)        # (col, row) to deploy on; rows 17-31 are the user's side
DRY_RUN = False       # True = only print the taps, don't send them


def ratio_box_centre(box):
    """Centre of a ratio box (x1, y1, x2, y2) in screen pixels."""
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2 * SCREEN_WIDTH), int((y1 + y2) / 2 * SCREEN_HEIGHT)


def card_slot_centre(slot):
    return ratio_box_centre(REGIONS[f"hand_{slot}"])


def tile_centre(col, row):
    """Centre of a grid tile in screen pixels (same grid as board_state.py)."""
    left, right = ARENA_LEFT * SCREEN_WIDTH, ARENA_RIGHT * SCREEN_WIDTH
    top, bottom = ARENA_TOP * SCREEN_HEIGHT, ARENA_BOTTOM * SCREEN_HEIGHT
    x = left + (col + 0.5) * (right - left) / GAME_COLUMNS
    y = top + (row + 0.5) * (bottom - top) / GAME_ROWS
    return int(x), int(y)


def adb_shell(command):
    if DRY_RUN:
        print(f"[dry run] adb shell {command}")
        return
    subprocess.run([ADB, "-s", DEVICE, "shell", command], check=True)


def play_card(slot, col, row):
    """Tap the card, then the tile -- sent as ONE adb command so it's a single round trip."""
    card_x, card_y = card_slot_centre(slot)
    tile_x, tile_y = tile_centre(col, row)
    print(f"slot {slot} at ({card_x}, {card_y})  ->  tile ({col}, {row}) at ({tile_x}, {tile_y})")
    adb_shell(f"input tap {card_x} {card_y}; input tap {tile_x} {tile_y}")


if __name__ == "__main__":
    slot, col, row = SLOT, *TILE
    if len(sys.argv) == 4:
        slot, col, row = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])

    start = time.perf_counter()
    play_card(slot, col, row)
    print(f"took {(time.perf_counter() - start) * 1000:.0f} ms")
