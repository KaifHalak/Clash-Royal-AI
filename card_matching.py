"""Match card images against the current deck or the full card library."""

import os
import shutil
import sys

import cv2

DEBUG_CROPS_DIR = "C:/MyStuff/ClashRoyal/debug-crops"

# Folders containing the complete card library and the cards seen in the
# current match.
ALL_CARDS_DIR = "C:/MyStuff/ClashRoyal/all-cards"
CURRENT_DECK_DIR = "C:/MyStuff/ClashRoyal/current-deck"
ADD_NEW_CARDS = "C:/MyStuff/ClashRoyal/temp"


# A match must be this confident before the card is accepted.
CONFIDENCE_THRESHOLD = 0.70
# Template matching requires both images to have the same dimensions.
MATCH_SIZE = (128, 128)

# ORB detector -- shared instance, reused across all comparisons
orb = cv2.ORB_create(nfeatures=500)
# Tune by testing how many keypoint matches count as "same card".
MIN_GOOD_MATCHES = 10
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


class CardMatcher:
    def __init__(self):
        # Load the full card library once, then load the smaller per-match
        # cache used for faster lookups.
        self.all_cards = self._load_dir(ALL_CARDS_DIR)
        self.current_deck = self._load_dir(CURRENT_DECK_DIR)
        
        # When we want to add new cards to the deck.
        self.new_cards_added = 0

    def _load_dir(self, path):
        """Load and resize every readable image in a folder."""
        refs = {}

        # Create missing folders so the matcher can be used on a new project.
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)
            return refs

        for fname in os.listdir(path):
            name = os.path.splitext(fname)[0]
            img = cv2.imread(os.path.join(path, fname))
            if img is not None:
                # Store resized templates to keep comparisons consistent.
                refs[name] = cv2.resize(img, MATCH_SIZE)

        # refs = {card_name: image instance as numpy array}
        return refs

    def _best_match_orb(self, crop, refs):
        """Return the reference with the highest ORB match score."""
        if not refs:
            return None, 0

        best_name, best_score = None, -1
        for name, ref in refs.items():
            score = orb_match_score(crop, ref)
            if score > best_score:
                best_name, best_score = name, score

        return best_name, best_score
    
    
    def _best_match(self, crop, refs):
        """Return the reference with the highest matchTemplate score."""
        if not refs:
            return None, -1

        best_name, best_score = None, -1
        for name, ref in refs.items():
            result = cv2.matchTemplate(crop, ref, cv2.TM_CCOEFF_NORMED)
            score = result.max()
            if score > best_score:
                best_name, best_score = name, score

        return best_name, best_score

    def _promote(self, name):
        """Copy a newly-confirmed card into current_deck (memory + disk)."""
        if name in self.current_deck:
            return

        self.current_deck[name] = self.all_cards[name]

        src = os.path.join(ALL_CARDS_DIR, f"{name}.png")
        dst = os.path.join(CURRENT_DECK_DIR, f"{name}.png")
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)

    def identify(self, crop):
        """
        Returns (card_name, score). Checks current_deck first (fast path),
        falls back to the full all_cards set only if that misses.
        """
        # Resize the incoming screenshot crop to match the stored templates.
        crop_new_res = cv2.resize(crop, MATCH_SIZE)
  
        # Fast path: known deck for this match
        name, score = self._best_match(crop_new_res, self.current_deck)
        if score > CONFIDENCE_THRESHOLD:
            return name, score

    
        # Fallback: full search, only reached for cards not yet seen this match
        name, score = self._best_match(crop_new_res, self.all_cards)
        if score > CONFIDENCE_THRESHOLD:
            self._promote(name)
            return name, score

        # Let the caller decide how to handle an uncertain result.
        # Low confidence means the caller should keep the last known state.
        
    
    
        return None, score

    
    def add_new_cards(self, crop):
        # TODO: Dedup does not work as intended. Need to fix
        
        # Resize the incoming screenshot crop to match the stored templates.
        crop_new_res = cv2.resize(crop, MATCH_SIZE)
        
        name, score = self._best_match(crop_new_res, self.current_deck)
    
        if score > CONFIDENCE_THRESHOLD:
            print("Same")
            return name, score

        
        
        new_card_name = f"new_card_{self.new_cards_added}"

        self.new_cards_added += 1
        
        self.current_deck[new_card_name] = crop_new_res
        
        os.makedirs(ADD_NEW_CARDS, exist_ok=True)
        crop_path = os.path.join(ADD_NEW_CARDS,f"{new_card_name}.png")
        cv2.imwrite(crop_path, crop)
        
        return new_card_name, score
        
        
        
        
    
    def reset_deck(self):
        """Call between matches so the previous game's deck doesn't leak in."""
        self.current_deck = {}
        self.new_cards_added = 0
        if os.path.isdir(CURRENT_DECK_DIR):
            shutil.rmtree(CURRENT_DECK_DIR)
        os.makedirs(CURRENT_DECK_DIR, exist_ok=True)


def orb_match_score(img1, img2):
    """Return the number of ORB keypoint matches between two images."""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    _, des1 = orb.detectAndCompute(gray1, None)
    _, des2 = orb.detectAndCompute(gray2, None)

    # No descriptors means one image has no usable keypoints.
    if des1 is None or des2 is None:
        return 0

    matches = bf.match(des1, des2)
    return len(matches)


# ==== DEBUGGING FUNCTIONS ====
def debug_resize_image(image_path, output_path="debug_resized.png"):
    """Resize a local image to MATCH_SIZE and save it for visual debugging."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not open image: {image_path}")

    resized_image = cv2.resize(image, MATCH_SIZE)

    if not cv2.imwrite(output_path, resized_image):
        raise OSError(f"Could not save debug image: {output_path}")

    return output_path


def debug_match_folder(folder_path, card_matcher):
    """Match every image in a folder and print the result for each file."""
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"Folder does not exist: {folder_path}")

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    results = []

    # Sort the names so repeated debugging runs process files in the same order.
    for filename in sorted(os.listdir(folder_path)):
        image_path = os.path.join(folder_path, filename)
        extension = os.path.splitext(filename)[1].lower()

        if not os.path.isfile(image_path) or extension not in image_extensions:
            continue

        image = cv2.imread(image_path)
        if image is None:
            print(f"{filename}: could not read image")
            continue

        card_name, score = card_matcher.identify(image)
        result = {
            "filename": filename,
            "card_name": card_name,
            "score": score,
        }
        results.append(result)

        match_name = card_name if card_name is not None else "NO MATCH"
        print(f"{filename}: {match_name} (score: {score:.4f})")

    return results


if __name__ == "__main__":
    # debug_resize_image("C:/MyStuff/ClashRoyal/debug-crops/hand_1.png", "debug_resized-1.png")
    # debug_resize_image("C:/MyStuff/ClashRoyal/all-cards/knight.png", "debug_resized-2.png")

    card_matcher = CardMatcher()
    # debug_match_folder("C:\\MyStuff\\ClashRoyal\\current-hand", card_matcher)
    
    image_path = "C:/MyStuff/ClashRoyal/debug-crops/hand_1.png"
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        sys.exit(1)
    
    card_name, score = card_matcher.identify(image)
    print(f"Debug match result for {image_path}: {card_name} (score: {score:.4f})")
    
