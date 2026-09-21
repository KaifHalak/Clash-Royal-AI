import os
import cv2


class ElixirReader:
    """
    Reads the current elixir count (0-10) from a cropped screenshot region
    using template matching against 11 known digit references (0-9, 10).

    Since the possible values are a fixed, closed set, we compare the live
    crop against every reference and pick whichever scores highest -- that
    score doubles as a confidence measure (0-100).
    """

    def __init__(
        self
    ):
        self.digit_dir = "C:\\MyStuff\\ClashRoyal\\elixir_digits"          
        self.review_dir = "C:\\MyStuff\\ClashRoyal\\elixir_review",# low-confidence reads get dumped here for manual review
        self.match_size = (60, 80)        # (width, height) -- common size everything gets resized to before comparing
        self.confidence_threshold = 40  # at/below this, save the crop for review

        os.makedirs(self.review_dir, exist_ok=True)

        self.digit_refs = {}     # name -> preprocessed reference image
        self.review_counter = 0  # used to give each saved review image a unique filename

        self._load_digit_refs()

    def _load_digit_refs(self):
        """Load and resize every reference template from disk into memory, once."""
        self.digit_refs = {}

        if not os.path.isdir(self.digit_dir):
            os.makedirs(self.digit_dir, exist_ok=True)
            return  # nothing to load yet -- caller needs to add reference images first

        for fname in os.listdir(self.digit_dir):
            name = os.path.splitext(fname)[0]  # filename without extension -> "0".."9" or "10"
            img = cv2.imread(os.path.join(self.digit_dir, fname), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                self.digit_refs[name] = cv2.resize(img, self.match_size)

    def _isolate_white_text(self, crop, sat_max=60, val_min=180):
        """
        Keep only near-white pixels (the digit) and discard everything else.
        """
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        mask = (s < sat_max) & (v > val_min)  # low saturation + high brightness = white text
        return (mask * 255).astype("uint8")

    def _remove_horizontal_line(self, thresh, line_len=75):
        """
        Strip out the elixir bar's tick line, which can cut across a digit.
        A long thin horizontal opening survives only on things at least
        `line_len` pixels wide in a row -- the line qualifies, digit strokes don't.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_len, 1))
        detected_line = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        return cv2.subtract(thresh, detected_line)

    def _preprocess(self, crop):
        """Raw elixir-region screenshot crop -> clean binary digit mask."""
        # Upscale first, while pixel data is still smooth -- avoids interpolation
        # artifacts that show up when resizing an already-binary mask.
        upscaled = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

        # Smooth out jagged/blocky pixel edges from the small source game font.
        blurred = cv2.GaussianBlur(upscaled, (3, 3), 0)

        thresh = self._isolate_white_text(blurred)
        thresh = self._remove_horizontal_line(thresh)

        # Slightly thicken strokes -- helps thin shapes like "0" and "1" survive cleanly.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        thresh = cv2.dilate(thresh, kernel, iterations=1)

        return thresh

    def read(self, crop, debug=False):
        """
        Returns (value, confidence).
        value: int 0-10, or None if no reference templates are loaded yet.
        confidence: 0-100, from the best matchTemplate score against all 11 digit refs.
        """
        thresh = self._preprocess(crop)
        thresh_resized = cv2.resize(thresh, self.match_size)

        if debug:
            cv2.imwrite("debug_elixir_crop.png", thresh)

        if not self.digit_refs:
            return None, 0  # no templates loaded -- nothing to compare against

        # Compare against every known digit, keep whichever scores highest.
        best_name, best_score = None, -1.0
        for name, ref in self.digit_refs.items():
            result = cv2.matchTemplate(thresh_resized, ref, cv2.TM_CCOEFF_NORMED)
            score = result.max()
            if score > best_score:
                best_name, best_score = name, score

        # TM_CCOEFF_NORMED ranges roughly -1 to 1 -- clamp negatives, scale to 0-100.
        confidence = int(max(0.0, best_score) * 100)

        # Low-confidence reads get saved so we can eyeball what's tripping up matching.
        if confidence <= self.confidence_threshold:
            review_path = os.path.join(
                self.review_dir,
                f"low_conf_{self.review_counter}_guess{best_name}_{confidence}.png",
            )
            cv2.imwrite(review_path, thresh)
            self.review_counter += 1

        return int(best_name), confidence