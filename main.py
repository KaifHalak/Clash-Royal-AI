from turtle import width

import cv2
import os

from window_capture import WindowCapture
from card_matching import CardMatcher
from static_elements_extractor import extract_static_elements
from elixir_detection_image_match import ElixirExtractor
from tower_health import TowerHealthExtractor



def main():
    window_capture = WindowCapture("LDPlayer").start()
    card_matcher = CardMatcher()
    elixir_reader = ElixirExtractor()
    tower_health_reader = TowerHealthExtractor()
    step = 0
    hand_res = {}
    for frame in window_capture.frames(fps=2):  # <-- control rate here, e.g. 2 frames/sec to LLM
        # frame is a BGR numpy array

        static_elements = extract_static_elements(frame)

    
        
        for name, element in static_elements.items():
            if name.startswith("hand"):
                card_name, score = card_matcher.identify(element)
                match_name = card_name if card_name is not None else "NO MATCH"
                hand_res[name] = match_name
                print(f"{name}: {match_name} (score: {score:.4f})")
    
            # if name.startswith("elixir"):
            #     elixir_value = elixir_reader.read(element, debug=False)
            #     print(f"{name}: {elixir_value}")
                
            # if name.startswith("tower_enemy"):
            #     tower_hp = tower_health_reader.read(element)
            #     print(f"{name}: {tower_hp}")
                
                # cv2.imwrite(f"./temp/debug_{name}.png", element)
                # step += 1

        # cv2.imshow("AI Vision", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            window_capture.stop()
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
