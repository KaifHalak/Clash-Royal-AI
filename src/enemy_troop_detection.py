import cv2
import numpy as np
from paddleocr import TextRecognition

#TODO: It takes the princess tower health as a badge. This needs to be fixed
#TODO: There are a bit of inaccuracies but for now its fine

#TODO: Get current health and card level

# ================= CONFIG =================
# Crop to only show arena
CROP_LEFT, CROP_RIGHT = 50, 50
CROP_TOP, CROP_BOTTOM = 70, 250

# Enemy badge inside colours (RGB). The badge is a gradient between these two,
# so we accept every colour BETWEEN them (+ tolerance), not just near each one.
BADGE_INSIDE_COLOURS_RGB = [(226, 38, 94), (172, 22, 67)]
COLOUR_TOLERANCE = 30          # +- per channel; raise to ~15 if badges are missed

# Badge border colour (RGB), OR'd into the colour mask. Tune with the tuner if the
# badge outline needs to be picked up separately from the inside gradient.
BORDER_COLOUR_RGB = (255, 255, 255)
BORDER_COLOUR_TOLERANCE = 20   # +- per channel

# Badge shape filters (tune after looking at the plots)
BADGE_MIN_SIDE = 8           # px, smallest badge side we accept
BADGE_MAX_SIDE = 24          # px, largest
BADGE_MIN_SQUARENESS = 0.7    # short side / long side (1.0 = perfect square)
BADGE_MIN_FILL = 0.5          # white area / box area, measured on the JOINED mask
CANDIDATE_JOIN_KERNEL = 3     # px; closes gaps so the badge's two sides become one candidate box
OPEN_KERNEL_SIZE = 2          # px; speck-removal kernel size, 0 = skip the OPEN step

# Damaged troop: HP bar attached on the right -> keep the left square only.
# A blob wider than (height * HP_BAR_WIDE_RATIO) is treated as badge+HP-bar.
HP_BAR_WIDE_RATIO = 1.3
BADGE_MAX_ASPECT = 3.0        # wider than height * this = a bar, not badge+HP-bar; clean_mask drops it

# Digit check: badge box must have a moderate amount of black inside (the digit), not too little or too much
DIGIT_MIN_BLACK_RATIO = 0.10  # at least 10% black inside the badge box (digit present)
DIGIT_MAX_BLACK_RATIO = 0.82  # at most 60% black (otherwise not a solid badge)


# ================= TROOP CROP =================
# NOTE:
# - Badge side and width are the same
# - Once a badge is detected, a box is drawn around the badge. This box has the lenght of the shorter side of the badge -> badge side / width

# Troop crop placement, in units of badge size (so it scales with resolution)
TROOP_CROP_SIZE_MULT = 6.0    # crop side = badge side * this

# Move the crop left/right
# >0 - move right, <0 - move left. 1.0 = badge width.
TROOP_CROP_SHIFT_X_MULT = 1.0 # badge width * this = move the crop horizontally by this much

# 0 starts the crop right at the badge's bottom edge.
# 0.5: go 10px down the badge. Start the crop there
# -0.5: go 10px up the badge. Start the crop there
# Note: 10px is just an example. The actual pixel is 0.5 * badge side
TROOP_CROP_GAP_Y_MULT = 0.0   # gap between badge bottom and crop top

# Enemy towers also have badges -> ignore them. Fractions of the cropped arena (x1, y1, x2, y2).
TOWER_IGNORE_REGIONS = []     # fill in after seeing where towers land


class TroopExtractor:
    def __init__(self, frame):
        """frame: full game frame (BGR numpy array, from cv2)."""
        assert frame is not None, "Frame is None"
        self.frame = frame

        # Instantiated once -- loading the model is expensive, predict() is not.
        # Same model/settings as the elixir and tower-health readers
        self.ocr = TextRecognition(model_name="PP-OCRv6_tiny_rec", enable_mkldnn=False)

    # ================= STEPS =================
    def crop_to_arena(self, image_bgr, crop_left=CROP_LEFT, crop_right=CROP_RIGHT,
                      crop_top=CROP_TOP, crop_bottom=CROP_BOTTOM):
        """Remove UI edges so only the arena remains."""
        full_height, full_width = image_bgr.shape[:2]
        return image_bgr[crop_top:full_height - crop_bottom, crop_left:full_width - crop_right]

    def build_badge_colour_mask(self, arena_bgr,
                                inside_colours_rgb=BADGE_INSIDE_COLOURS_RGB,
                                colour_tolerance=COLOUR_TOLERANCE,
                                border_colour_rgb=BORDER_COLOUR_RGB,
                                border_colour_tolerance=BORDER_COLOUR_TOLERANCE,
                                tower_ignore_regions=TOWER_IGNORE_REGIONS):
        """White where the pixel colour falls inside the badge colours OR the border colour.
            We are using thresholding technique. The reason its called "mask" bcs in the next step, we will use this to create the boxes around the badges in the orignial image using masking.


        """

        # Inside-colour mask: every colour between the two gradient endpoints (+ tolerance)
        inside_colours_bgr = np.array([(b, g, r) for r, g, b in inside_colours_rgb], dtype=np.int16)
        inside_lower = np.clip(inside_colours_bgr.min(axis=0) - colour_tolerance, 0, 255).astype(np.uint8)
        inside_upper = np.clip(inside_colours_bgr.max(axis=0) + colour_tolerance, 0, 255).astype(np.uint8)
        inside_mask = cv2.inRange(arena_bgr, inside_lower, inside_upper)

        # This will make the badge color white and the number inside (and everything else) black
        # cv2.imshow("Inside Mask", inside_mask)

        # Border-colour mask: badge outline colour, OR'd in separately
        border_b, border_g, border_r = border_colour_rgb[2], border_colour_rgb[1], border_colour_rgb[0]
        border_colour_bgr = np.array([border_b, border_g, border_r], dtype=np.int16)
        border_lower = np.clip(border_colour_bgr - border_colour_tolerance, 0, 255).astype(np.uint8)
        border_upper = np.clip(border_colour_bgr + border_colour_tolerance, 0, 255).astype(np.uint8)
        border_mask = cv2.inRange(arena_bgr, border_lower, border_upper)

        # This will make the numbers white and everything else black
        # cv2.imshow("Border Mask", border_mask)


        # Combine the 2 masks to show the numbers and the badges as white and everything else as black
        joined_mask = cv2.bitwise_or(inside_mask, border_mask)

        #TODO: For now we can ignore this as the tower badge does not have a digit inside, so its not detected by default. We will leave the code for future use
        # Blank out tower areas so tower badges aren't treated as troops
        arena_height, arena_width = joined_mask.shape
        for x1, y1, x2, y2 in tower_ignore_regions:
            joined_mask[int(y1 * arena_height):int(y2 * arena_height),
                        int(x1 * arena_width):int(x2 * arena_width)] = 0
        return (inside_mask, border_mask, joined_mask)

    def has_digit_inside(self, cleaned_mask, x, y, side,
                         digit_min_black_ratio=DIGIT_MIN_BLACK_RATIO,
                         digit_max_black_ratio=DIGIT_MAX_BLACK_RATIO):
        """Badge test: the badge box has a moderate amount of black inside (the digit)."""
        badge_region = cleaned_mask[y:y + side, x:x + side] > 0          # True = white
        black_ratio = 1 - badge_region.mean()                            # fraction of the box that's black

        return digit_min_black_ratio <= black_ratio <= digit_max_black_ratio

    def find_badges(self, cleaned_mask,
                    inside_mask,
                    candidate_join_kernel=CANDIDATE_JOIN_KERNEL,
                    badge_min_side=BADGE_MIN_SIDE,
                    badge_max_side=BADGE_MAX_SIDE,
                    badge_min_squareness=BADGE_MIN_SQUARENESS,
                    badge_min_fill=BADGE_MIN_FILL,
                    hp_bar_wide_ratio=HP_BAR_WIDE_RATIO,
                    digit_min_black_ratio=DIGIT_MIN_BLACK_RATIO,
                    digit_max_black_ratio=DIGIT_MAX_BLACK_RATIO,
                    verbose=False):
        """Return (accepted_badges, rejected_candidates).

        accepted_badges: list of (x, y, side)
        rejected_candidates: list of (x, y, side, reason, measured_value)
        """
        candidate_join_kernel = max(1, candidate_join_kernel)  # guard: kernel must be >= 1

        # Step 1: join the badge's pieces into one blob, ONLY to find candidate boxes
        join_kernel = np.ones((candidate_join_kernel, candidate_join_kernel), np.uint8)
        joined_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, join_kernel)


        # This gives us information abt each blob:
            # num_labels: number of blobs (including background)
            # blob_stats: information for each blob
        num_labels, _, blob_stats, _ = cv2.connectedComponentsWithStats(joined_mask)


        #For now, we make a big blob for each badge to check if it qualifies as a badge
        # cv2.imshow("Joined Mask", joined_mask)

        accepted_badges = []
        rejected_candidates = []
        for blob_index in range(1, num_labels):                          # 0 = background, so we ignore

            #blob information
            x, y, width, height, area = blob_stats[blob_index]

            fill_ratio = area / (width * height)

            # Damaged troop: HP bar attached on the right -> keep the left square only
            # We are assuming that the badges are going to be squares
            if width > height * hp_bar_wide_ratio:
                width = height

            badge_side = min(width, height)
            squareness = min(width, height) / max(width, height)

            # Step 2: shape filters
            if not (badge_min_side <= badge_side <= badge_max_side):
                if verbose:
                    print(f"REJECT size   at ({x},{y}) side={badge_side}")
                rejected_candidates.append((x, y, badge_side, "size", badge_side))
                continue
            if squareness < badge_min_squareness:
                if verbose:
                    print(f"REJECT square at ({x},{y}) squareness={squareness:.2f}")
                rejected_candidates.append((x, y, badge_side, "square", squareness))
                continue

            if fill_ratio < badge_min_fill:
                if verbose:
                    print(f"REJECT fill   at ({x},{y}) fill={fill_ratio:.2f}")
                rejected_candidates.append((x, y, badge_side, "fill", fill_ratio))
                continue


            #  === Whatever passes through above means its a valid badge ===

            # Step 3: digit check on the UN-joined mask (joining would hide the black centre)

            # We r cropping the mask to only show the badge
            # The reason we r using inside_mask bcs we want the number inside the badge to be black for the ratio to be accurate
            badge_region = inside_mask[y:y + badge_side, x:x + badge_side] > 0

            # .mean() gives the fraction of white pixels. To find fraction of black pixels, we do 1 - mean()
            black_ratio = 1 - badge_region.mean()

            if not self.has_digit_inside(inside_mask, x, y, badge_side,
                                         digit_min_black_ratio, digit_max_black_ratio):
                if verbose:
                    print(f"REJECT digit  at ({x},{y}) black_ratio={black_ratio:.2f}")
                rejected_candidates.append((x, y, badge_side, "digit", black_ratio))
                continue

            accepted_badges.append((x, y, badge_side))


        for badge in accepted_badges:
            x, y, side = badge
            print(f"ACCEPTED badge at ({x},{y}) side={side}")

        return accepted_badges, rejected_candidates

    def get_troop_crop_box(self, badge, arena_width, arena_height,
                           troop_crop_size_mult=TROOP_CROP_SIZE_MULT,
                           troop_crop_shift_x_mult=TROOP_CROP_SHIFT_X_MULT,
                           troop_crop_gap_y_mult=TROOP_CROP_GAP_Y_MULT):
        """Square box below the badge where the troop body is, clipped to the image."""
        badge_x, badge_y, badge_side = badge
        crop_side = int(badge_side * troop_crop_size_mult)
        crop_centre_x = badge_x + badge_side / 2 + badge_side * troop_crop_shift_x_mult
        crop_top = badge_y + badge_side + badge_side * troop_crop_gap_y_mult

        x1 = int(max(0, crop_centre_x - crop_side / 2))
        y1 = int(max(0, crop_top))
        x2 = int(min(arena_width, x1 + crop_side))
        y2 = int(min(arena_height, y1 + crop_side))
        return x1, y1, x2, y2

    def read_badge_number(self, arena_bgr, badge, scale=8, pad=4):
        """OCR the digit inside one badge. Returns (digits, confidence 0-100); digits is '' if no digit was read."""
        badge_x, badge_y, badge_side = badge
        badge_crop = arena_bgr[badge_y:badge_y + badge_side, badge_x:badge_x + badge_side]

        # The digit is only a few pixels tall, so upscale it, and pad so it isn't touching the image edge
        badge_crop = cv2.resize(badge_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        badge_crop = cv2.copyMakeBorder(badge_crop, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

        for result in self.ocr.predict(badge_crop):
            # The model can't be restricted to digits, so drop any non-digit characters from its output
            digits = "".join(character for character in result["rec_text"] if character.isdigit())
            return digits, int(result.get("rec_score", 0) * 100)
        return "", 0

    # ================= RUN =================
    def extract_troop_crops(self):
        """Return the list of troop crops (BGR numpy arrays), one per detected enemy badge."""
        arena_bgr = self.crop_to_arena(self.frame)
        arena_height, arena_width = arena_bgr.shape[:2]

        badge_inside_mask, badge_border_mask, badge_joined_mask = self.build_badge_colour_mask(arena_bgr)
        detected_badges, rejected_badges = self.find_badges(badge_joined_mask, badge_inside_mask)
        print(f"Detected {len(detected_badges)} enemy badges")

        troop_crops = []

        for badge_number, badge in enumerate(detected_badges):
            badge_x, badge_y, badge_side = badge
            badge_text, badge_confidence = self.read_badge_number(arena_bgr, badge)
            print(f"Badge {badge_number} at ({badge_x},{badge_y}): '{badge_text}' ({badge_confidence}%)")
            x1, y1, x2, y2 = self.get_troop_crop_box(badge, arena_width, arena_height)

            troop_crop = arena_bgr[y1:y2, x1:x2]
            troop_crops.append(troop_crop)

        return troop_crops
