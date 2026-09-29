"""
Reads the current elixir count (0-10) from a cropped screenshot region.

Switched from template matching (11 known digit references) to PaddleOCR's
standalone TextRecognition model, matching the approach used for tower HP --
same upscale/binarize preprocessing pipeline, same engine. Line removal is
kept since it's specific to the elixir bar's tick mark bleeding into the
crop; blob removal (used for tower HP's stray UI fragments) isn't needed
here since the elixir crop doesn't have that issue.
"""

import os

import cv2
import numpy as np
from PIL import Image
from paddleocr import TextRecognition


class ElixirExtractor:
    def __init__(self):
        # Instantiated once -- loading the model is expensive, predict() is not.
        self.ocr = TextRecognition(
            model_name="PP-OCRv6_tiny_rec",
            enable_mkldnn=False,  # works around a known CPU oneDNN backend bug
        )

        self.review_dir = "C:\\MyStuff\\ClashRoyal\\elixir_review"  # low-confidence reads saved here for manual review
        self.threshold = 200          # binarize cutoff -- tuned by eye against sample crops
        self.scale = 8               # upscale factor before binarizing
        self.line_len = 75            # elixir bar tick-line removal kernel width
        self.confidence_threshold = 40  # 0-100 scale; at/below this, save the crop for review

        os.makedirs(self.review_dir, exist_ok=True)
        self.review_counter = 0  # used to give each saved review image a unique filename

    def _preprocess(self, crop):
        """Raw BGR crop (numpy array, from cv2) -> cleaned binary digit mask."""
        # cv2 gives BGR numpy arrays; PIL/Image.resize wants RGB PIL images.
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        # Upscale with LANCZOS resampling, done BEFORE binarizing so the
        # resample has real gradient data to work with.
        width, height = image.size
        image = image.resize((width * self.scale, height * self.scale), Image.LANCZOS)

        image = image.convert("L")  # grayscale
        image = image.point(lambda value: 0 if value < self.threshold else 255, "1")  # binarize

        arr = np.array(image.convert("L"))
        

        return arr

    def read(self, crop, debug=False):
        """
        Returns (value, confidence).
        value: int 0-10, or None if nothing readable was found.
        confidence: 0-100, from PaddleOCR's recognition score.
        """
        thresh = self._preprocess(crop)

        if debug:
            cv2.imwrite("debug_elixir_crop.png", thresh)

        # PaddleOCR expects a 3-channel (BGR) array, not single-channel binary.
        bgr_arr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        result = self.ocr.predict(bgr_arr)

        value, confidence = None, 0
        for res in result:
            text = res["rec_text"]
            score = res.get("rec_score", 0)  # NOTE: key unconfirmed -- verify against actual output
            digits = "".join(c for c in text if c.isdigit())
            if digits:
                value = int(digits)
            confidence = int(score * 100)
            break

        if confidence <= self.confidence_threshold:
            review_path = os.path.join(
                self.review_dir,
                f"low_conf_{self.review_counter}_guess{value}_{confidence}.png",
            )
            cv2.imwrite(review_path, thresh)
            self.review_counter += 1

        return value, confidence