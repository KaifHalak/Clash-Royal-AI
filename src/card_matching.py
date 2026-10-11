"""Match card images against a hard-coded deck, using PaddleX's
PP-ShiTuV2_rec image-embedding model instead of matchTemplate/ORB.
"""

import os

import cv2
import numpy as np
from paddlex import create_model
from static_elements_extractor import extract_static_elements

# The deck being played. Each name is a filename (without extension) in ALL_CARDS_DIR.
# Only these cards are matched against, which cuts down on false positives.
MY_DECK = [
    "arrows",
    "fireball",
    "giant",
    "knight",
    "archers",
    "mini_pekka",
    "minions",
    "musketeer",
]

ALL_CARDS_DIR = "C:/MyStuff/ClashRoyal/all-cards"
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

        # Embeddings are cached at load time, NOT recomputed on every
        # identify() call.
        # {card_name: embedding vector}
        self.deck_embeddings = self._load_and_embed_deck(MY_DECK)

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

    def _load_and_embed_deck(self, deck):
        """Embed each deck card's image from ALL_CARDS_DIR, return {card_name: embedding}."""
        file_paths = []
        for name in deck:
            matches = [
                os.path.join(ALL_CARDS_DIR, name + ext)
                for ext in IMAGE_EXTENSIONS
                if os.path.isfile(os.path.join(ALL_CARDS_DIR, name + ext))
            ]
            if not matches:
                raise FileNotFoundError(f"No image for deck card '{name}' in {ALL_CARDS_DIR}")
            file_paths.append(matches[0])

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

    # ---- public API ----

    def identify(self, crop):
        """Returns (card_name, score), matching only against MY_DECK."""
        crop_embedding = self._embed_crop(crop)

        name, score = self._best_match(crop_embedding, self.deck_embeddings)
        if score > CONFIDENCE_THRESHOLD:
            return name, score

        # Low confidence -- caller should keep the last known state.
        return None, score

    # def add_new_cards(self, crop):
    #     """
    #     Used to seed all_cards from real gameplay. Checks only
    #     against the deck (not the full library) -- if no match, saves
    #     the crop as a new candidate reference and caches its embedding so
    #     the same card isn't added twice.
    #     """
    #     crop_embedding = self._embed_crop(crop)

    #     name, score = self._best_match(crop_embedding, self.deck_embeddings)
    #     if score > CONFIDENCE_THRESHOLD:
    #         print(f"Same: {name} ({score:.4f})")
    #         return name, score

    #     new_card_name = f"new_card_{self.new_cards_added}"
    #     self.new_cards_added += 1

    #     self.deck_embeddings[new_card_name] = crop_embedding

    #     os.makedirs(ADD_NEW_CARDS_DIR, exist_ok=True)
    #     crop_path = os.path.join(ADD_NEW_CARDS_DIR, f"{new_card_name}.png")
    #     cv2.imwrite(crop_path, crop)

    #     print(f"New card, score was {score:.4f}")
    #     return new_card_name, score


if __name__ == "__main__":
    frame = cv2.imread("C:/MyStuff/ClashRoyal/game_frames/frame_122.png")
    out = extract_static_elements(frame)
    
    hand1 = out["hand_1"]
    hand2 = out["hand_2"]
    hand3 = out["hand_3"]
    hand4 = out["hand_4"]
    next = out["next_card"]
    
    card_matcher = CardMatcher()
    name1, score1 = card_matcher.identify(hand1)
    name2, score2 = card_matcher.identify(hand2)
    name3, score3 = card_matcher.identify(hand3)
    name4, score4 = card_matcher.identify(hand4)
    
    name5, score5 = card_matcher.identify(next)

    
    print(f"Identified card: {name1} (score: {score1:.4f})")
    print(f"Identified card: {name2} (score: {score2:.4f})")
    print(f"Identified card: {name3} (score: {score3:.4f})")
    print(f"Identified card: {name4} (score: {score4:.4f})")
    print(f"Identified card: {name5} (score: {score5:.4f})")
    
    # =================================================
    # frame1 = cv2.imread("C:/MyStuff/ClashRoyal/game_frames/frame_122.png")
    # frame2 = cv2.imread("C:/MyStuff/ClashRoyal/game_frames/frame_122.png")
    
    # out1 = extract_static_elements(frame1)
    # out2 = extract_static_elements(frame2)
    
    # hand1_1 = out1["hand_1"]
    # hand2_1 = cv2.imread("C:/MyStuff/ClashRoyal/all-cards/giant.png")
    
    # card_matcher = CardMatcher()
    
    # card1_vec = card_matcher._embed_crop(hand1_1)
    # card2_vec = card_matcher._embed_crop(hand2_1)
    
    # score = card_matcher._cosine_similarity(card1_vec, card2_vec)
    # print(f"Cosine similarity between two identical cards: {score:.4f}")

    pass