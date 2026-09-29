"""Match card images against the current deck or the full card library,
using PaddleX's PP-ShiTuV2_rec image-embedding model instead of
matchTemplate/ORB.
"""

import os
import shutil

import cv2
import numpy as np
from paddlex import create_model

ALL_CARDS_DIR = "C:/MyStuff/ClashRoyal/all-cards"
CURRENT_DECK_DIR = "C:/MyStuff/ClashRoyal/current-deck"
# ADD_NEW_CARDS_DIR = "C:/MyStuff/ClashRoyal/temp"

# Cosine similarity threshold -- tune by testing against known same/different pairs.
CONFIDENCE_THRESHOLD = 0.35 # We can trust the semantic embedding. We still need a confidence threshold to avoid false positives.

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# No measured speedup past 5 for this workload -- keep it there rather than
# guessing a bigger number "for safety".
BATCH_SIZE = 5


class CardMatcher:
    def __init__(self):
        self.model = create_model("PP-ShiTuV2_rec")

        # Embeddings are cached at load/promote time, NOT recomputed on every
        # identify() call -- that's what keeps current_deck lookups fast.
        # {card_name: embedding vector}
        self.all_cards_embeddings = self._load_and_embed_dir(ALL_CARDS_DIR)
        self.current_deck_embeddings = self._load_and_embed_dir(CURRENT_DECK_DIR)

        self.new_cards_added = 0

    # ---- embedding helpers ----

    def _embed_paths(self, image_paths):
        """
        Runs the model on a list of image paths, returns {path: embedding}.
        Handles the known PaddleX batching bug: each result's "feature"
        field returns the whole batch's (batch_size, 512) array unsliced,
        not just that image's row -- so we index it ourselves using each
        result's position in the input list.
        """
        results = list(self.model.predict(image_paths, batch_size=BATCH_SIZE))

        embeddings = {}
        for i, res in enumerate(results):
            path = res["input_path"]
            feat_matrix = np.array(res["feature"])
            row = i % BATCH_SIZE
            embeddings[path] = feat_matrix[row]

        return embeddings

    def _embed_crop(self, crop):
        """Embeds a single in-memory BGR crop (numpy array), returns one vector."""
        results = list(self.model.predict(crop, batch_size=1))
        return np.array(results[0]["feature"])[0]

    def _load_and_embed_dir(self, path):
        """Load every image in a folder and return {card_name: embedding}."""
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)
            return {}

        file_paths = [
            os.path.join(path, fname)
            for fname in os.listdir(path)
            if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS
        ]
        if not file_paths:
            return {}

        path_embeddings = self._embed_paths(file_paths)

        # Re-key by card name (filename without extension) instead of full path.
        return {
            os.path.splitext(os.path.basename(p))[0]: emb
            for p, emb in path_embeddings.items()
        }

    @staticmethod
    def _cosine_similarity(vec_a, vec_b):
        return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

    def _best_match(self, crop_embedding, ref_embeddings):
        """Return the reference name with the highest cosine similarity."""
        if not ref_embeddings:
            return None, -1.0

        best_name, best_score = None, -1.0
        for name, ref_emb in ref_embeddings.items():
            score = self._cosine_similarity(crop_embedding, ref_emb)
            if score > best_score:
                best_name, best_score = name, score

        return best_name, best_score

    def _promote(self, name):
        """Copy a newly-confirmed card into current_deck (memory + disk)."""
        if name in self.current_deck_embeddings:
            return

        # Reuse the already-computed embedding -- no need to re-run the model.
        self.current_deck_embeddings[name] = self.all_cards_embeddings[name]

        src = os.path.join(ALL_CARDS_DIR, f"{name}.png")
        dst = os.path.join(CURRENT_DECK_DIR, f"{name}.png")
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)

    # ---- public API ----

    def identify(self, crop):
        """
        Returns (card_name, score). Checks current_deck first (fast path),
        falls back to the full all_cards set only if that misses.
        """
        crop_embedding = self._embed_crop(crop)

        # Fast path: known deck for this match.
        name, score = self._best_match(crop_embedding, self.current_deck_embeddings)
        if score > CONFIDENCE_THRESHOLD:
            return name, score

        # Fallback: full search, only reached for cards not yet seen this match.
        name, score = self._best_match(crop_embedding, self.all_cards_embeddings)
        if score > CONFIDENCE_THRESHOLD:
            self._promote(name)
            return name, score

        # Low confidence -- caller should keep the last known state.
        return None, score

    # def add_new_cards(self, crop):
    #     """
    #     Used to seed all_cards/current_deck from real gameplay. Checks only
    #     against current_deck (not the full library) -- if no match, saves
    #     the crop as a new candidate reference and caches its embedding so
    #     the same card isn't added twice.
    #     """
    #     crop_embedding = self._embed_crop(crop)

    #     name, score = self._best_match(crop_embedding, self.current_deck_embeddings)
    #     if score > CONFIDENCE_THRESHOLD:
    #         print(f"Same: {name} ({score:.4f})")
    #         return name, score

    #     new_card_name = f"new_card_{self.new_cards_added}"
    #     self.new_cards_added += 1

    #     self.current_deck_embeddings[new_card_name] = crop_embedding

    #     os.makedirs(ADD_NEW_CARDS_DIR, exist_ok=True)
    #     crop_path = os.path.join(ADD_NEW_CARDS_DIR, f"{new_card_name}.png")
    #     cv2.imwrite(crop_path, crop)

    #     print(f"New card, score was {score:.4f}")
    #     return new_card_name, score

    def reset_deck(self):
        """Call between matches so the previous game's deck doesn't leak in."""
        self.current_deck_embeddings = {}
        self.new_cards_added = 0
        if os.path.isdir(CURRENT_DECK_DIR):
            shutil.rmtree(CURRENT_DECK_DIR)
        os.makedirs(CURRENT_DECK_DIR, exist_ok=True)