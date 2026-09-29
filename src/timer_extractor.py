"""
Reads the tower HP number from a cropped screenshot region via OCR.

Unlike elixir (a closed 0-10 set matched against templates), tower HP can
be any 1-4 digit number -- too many possibilities to template-match, so
this uses OCR with a preprocessing pipeline tuned for the game's small
stylized digit font (upscale + LANCZOS, grayscale, binarize, strip stray
UI fragments).

Switched from Tesseract (~70% accuracy) to PaddleOCR's standalone
TextRecognition model, which reads the same stylized font more reliably
and faster -- no text-detection stage needed since the crop already
contains exactly one isolated number.

Known open issue -- see discussion, not code: handling what happens once
a tower is destroyed and the crop goes blank.
"""

import cv2
import numpy as np
from PIL import Image
from paddleocr import TextRecognition


class TimeExtractor:
    def __init__(self):
        # Instantiated once -- loading the model is expensive, predict() is not.
        self.ocr = TextRecognition(
            model_name="PP-OCRv6_tiny_rec",
            enable_mkldnn=False,  # works around a known CPU oneDNN backend bug
        )

        self.threshold = 200          # binarize cutoff -- tuned by eye against sample crops
        self.scale = 8                # upscale factor before binarizing
        self.min_area_ratio = 0.18    # blob-removal size cutoff, relative to the largest component

    def _preprocess(self, crop):
        """
        Raw BGR crop (numpy array, from cv2) -> cleaned, upscaled PIL image
        ready for OCR.
        """
        # cv2 gives BGR numpy arrays; PIL/Image.resize wants RGB PIL images.
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        # Upscale with LANCZOS resampling -- smoother than plain interpolation,
        # done BEFORE binarizing so the resample has real gradient data to work with.
        width, height = image.size
        image = image.resize((width * self.scale, height * self.scale), Image.LANCZOS)

        image = image.convert("L")  # grayscale
        image = image.point(lambda value: 0 if value < self.threshold else 255, "1")  # binarize

        image = self._remove_small_blobs(image)

        return image
    def _remove_small_blobs(self, image):
        """
        Removes small disconnected white blobs (e.g. stray UI fragments,
        crown icon corners) that bleed into the crop -- keeps only
        components whose area is at least min_area_ratio of the largest
        component's area.
        """
        arr = np.array(image.convert("L"))
        _, binary = cv2.threshold(arr, 127, 255, cv2.THRESH_BINARY)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num_labels <= 1:
            return image  # nothing found, unchanged

        areas = stats[1:, cv2.CC_STAT_AREA]  # skip label 0 (background)
        max_area = areas.max()

        cleaned = np.zeros_like(binary)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= max_area * self.min_area_ratio:
                cleaned[labels == i] = 255

        return Image.fromarray(cleaned)

    def read(self, crop):
        """
        Returns the timer as {"min": int, "sec": int}, or None if fewer
        than 3 digits were read (e.g. "123" -> 1 min 23 sec).
        `crop` is a BGR numpy array (e.g. from cv2 / your capture pipeline).
        """
        processed = self._preprocess(crop)

        # PaddleOCR expects a 3-channel (BGR) array, not single-channel binary.
        gray_arr = np.array(processed.convert("L"))
        bgr_arr = cv2.cvtColor(gray_arr, cv2.COLOR_GRAY2BGR)

        # cv2.imwrite("./temp/debug_timer.png", bgr_arr)  # Save the processed image for debugging

        result = self.ocr.predict(bgr_arr)

        for res in result:
            text = res["rec_text"]
            digits = "".join(c for c in text if c.isdigit())

            if len(digits) < 3:
                return None

            return {"min": int(digits[:-2]), "sec": int(digits[-2:])}

        return None