from turtle import width

import cv2
import os

from window_capture import WindowCapture
from card_matching import CardMatcher
from static_elements_extractor import extract_static_elements




def main():
    window_capture = WindowCapture("LDPlayer").start()
    card_matcher = CardMatcher()

    for frame in window_capture.frames(fps=2):  # <-- control rate here, e.g. 2 frames/sec to LLM
        # frame is a BGR numpy array

        static_elements = extract_static_elements(frame)

    

        for name, element in static_elements.items():
            if name.startswith("hand"):
                card_name, score = card_matcher.identify(element)
                match_name = card_name if card_name is not None else "NO MATCH"
                print(f"{name}: {match_name} (score: {score:.4f})")
    
               

        

        # cv2.imshow("AI Vision", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            window_capture.stop()
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
