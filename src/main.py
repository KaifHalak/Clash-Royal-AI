from turtle import width
import numpy as np
import cv2
import os
# from ultralytics import YOLO


from window_capture import WindowCapture
from static_elements_extractor import extract_static_elements
# from card_matching import CardMatcher
# from elixir_detection_ocr import ElixirExtractor
# from timer_extractor import TimeExtractor
# from tower_tracker import TowerTracker

 
FOLDER = "C:/MyStuff/ClashRoyal/game_frames"

def main():
    window_capture = WindowCapture("LDPlayer").start()
    # card_matcher = CardMatcher()
    # elixir_reader = ElixirExtractor()
    # timer_reader = TimeExtractor()
    # tower_tracker = TowerTracker()
    step = 0
    hand_res = {}

    for frame in window_capture.frames(fps=0.25): 

        step += 1
        static_elements = extract_static_elements(frame)
        cv2.imwrite(f"{FOLDER}/frame_{step}.png", frame)
                
        for name, element in static_elements.items():
            pass
            

            
            # cv2.imwrite(f"./temp/debug_{name}.png", element)
            
            # if name.startswith("hand"):
            #     card_name, score = card_matcher.identify(element)
            #     match_name = card_name if card_name is not None else "NO MATCH"
            #     hand_res[name] = match_name
            #     print(f"{name}: {match_name} (score: {score:.4f})")
    
            # if name.startswith("elixir"):
            #     elixir_value = elixir_reader.read(element, debug=False)
            #     print(f"{name}: {elixir_value}")
                
            # if name.endswith("king"):
            #     tower_hp = tower_tracker.update(name, element)
            #     print(f"{name}: {tower_hp}")
            #     print(f"{name}: {tower_tracker.is_king_active(name)}")
            #     cv2.imwrite(f"./temp/debug_{name}.png", element)
               
                # if (tower_tracker.is_dead(name)):
                #     print(f"{name}: DEAD")
                # else:
                #     print(f"{name}: {tower_hp}")
                
            #     cv2.imwrite(f"./temp/debug_{name}.png", element)
            #     step += 1
                
            # if (name.startswith("timer")):
                # cv2.imwrite(f"./temp/debug_{name}.png", element)
                # timer_value = timer_reader.read(element)
                # print(f"{timer_value}") 
                # cv2.imwrite(f"./temp/debug_{name}.png", element)
                # step += 1

        

        # cv2.imshow("AI Vision", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            window_capture.stop()
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()



 
