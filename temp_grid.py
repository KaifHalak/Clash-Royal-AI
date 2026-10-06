import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Draws the game's 18x32 tile grid on a saved frame so we can check how well it lines up.
# Use the matplotlib toolbar to zoom (magnifier) and pan (arrows); "home" resets the view.
# The bottom-right of the window shows the pixel and the (col, row) tile under the mouse.

# ================= CONFIG =================
FRAME_PATH = "C:/MyStuff/ClashRoyal/game_frames/frame_150.png"

# Measured on a 900x1600 screenshot. Stored as ratios so they work at any frame size.
REF_WIDTH, REF_HEIGHT = 900, 1600
ARENA_LEFT, ARENA_RIGHT = 70 / REF_WIDTH, 830 / REF_WIDTH
ARENA_TOP, ARENA_BOTTOM = 155 / REF_HEIGHT, 1190 / REF_HEIGHT
ENEMY_BRIDGE_START = 640 / REF_HEIGHT     # top edge of the river
USER_BRIDGE_START = 720 / REF_HEIGHT      # bottom edge of the river

GAME_COLUMNS, GAME_ROWS = 18, 32          # the game's tile grid

GRID_COLOUR = "yellow"
EDGE_COLOUR = "red"       # the measured arena + river edges
RIVER_COLOUR = "blue"


if __name__ == "__main__":
    frame = cv2.imread(FRAME_PATH)
    assert frame is not None, "Check FRAME_PATH"
    frame_height, frame_width = frame.shape[:2]

    # Ratios -> pixels for this frame
    left, right = ARENA_LEFT * frame_width, ARENA_RIGHT * frame_width
    top, bottom = ARENA_TOP * frame_height, ARENA_BOTTOM * frame_height
    river_top, river_bottom = ENEMY_BRIDGE_START * frame_height, USER_BRIDGE_START * frame_height

    tile_width = (right - left) / GAME_COLUMNS
    tile_height = (bottom - top) / GAME_ROWS
    print(f"{GAME_COLUMNS} x {GAME_ROWS} tiles, tile {tile_width:.1f} x {tile_height:.1f} px")

    figure, axis = plt.subplots(figsize=(7, 12))
    # nearest = real pixels show as blocks when zoomed in, instead of being blurred
    axis.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), interpolation="nearest")

    # River band
    axis.add_patch(Rectangle((left, river_top), right - left, river_bottom - river_top,
                             color=RIVER_COLOUR, alpha=0.2))

    # Tile grid (vector lines, so they stay thin at any zoom)
    for col in range(GAME_COLUMNS + 1):
        axis.vlines(left + col * tile_width, top, bottom, colors=GRID_COLOUR, linewidth=0.6)
    for row in range(GAME_ROWS + 1):
        axis.hlines(top + row * tile_height, left, right, colors=GRID_COLOUR, linewidth=0.6)

    # Measured edges on top, so we can see where the grid drifts from them
    for y in (top, river_top, river_bottom, bottom):
        axis.hlines(y, left, right, colors=EDGE_COLOUR, linewidth=1)
    for x in (left, right):
        axis.vlines(x, top, bottom, colors=EDGE_COLOUR, linewidth=1)

    # Hover readout: pixel + tile under the mouse
    def format_coord(x, y):
        if left <= x < right and top <= y < bottom:
            col = int((x - left) // tile_width)
            row = int((y - top) // tile_height)
            return f"px ({x:.0f}, {y:.0f})   tile ({col}, {row})"
        return f"px ({x:.0f}, {y:.0f})   outside arena"
    axis.format_coord = format_coord

    axis.set_title(f"{GAME_COLUMNS}x{GAME_ROWS} tiles")
    axis.axis("off")
    plt.tight_layout()
    plt.show()
