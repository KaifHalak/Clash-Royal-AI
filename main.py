from turtle import width

import cv2
import os

from window_capture import WindowCapture
from card_matching import CardMatcher
from static_elements_extractor import extract_static_elements

DEBUG_CROPS_DIR = "C:/MyStuff/ClashRoyal/debug-crops"


def main():
    window_capture = WindowCapture("LDPlayer").start()
    card_matcher = CardMatcher()

    for frame in window_capture.frames(
        fps=2
    ):  # <-- control rate here, e.g. 2 frames/sec to LLM
        # frame is a BGR numpy array

        cv2.imshow("AI Vision", frame)

        height, width, channels = frame.shape
        # print(f"Width: {width}, Height: {height}")

        elements = extract_static_elements(frame)

        debug_crop_index = 0

        for name, element in elements.items():
            if name.startswith("hand"):
                # card_name, score = card_matcher.identify(element)
                # match_name = card_name if card_name is not None else "NO MATCH"
                # print(f"{name}: {match_name} (score: {score:.4f})")

                # Save each resized crop for debugging and visual inspection.
                os.makedirs(DEBUG_CROPS_DIR, exist_ok=True)
                crop_path = os.path.join(
                    DEBUG_CROPS_DIR,
                    f"{name}.png",
                )
                cv2.imwrite(crop_path, element)

        debug_crop_index += 1

        # cv2.imshow("AI Vision", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            window_capture.stop()
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
