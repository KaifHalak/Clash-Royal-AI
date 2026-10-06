from turtle import width
import numpy as np
import cv2
# from ultralytics import YOLO

import os
import time

from troop_detector import TroopDetector
from board_state import BoardState


from window_capture import WindowCapture
from static_elements_extractor import extract_static_elements
# from card_matching import CardMatcher
# from elixir_detection_ocr import ElixirExtractor
# from timer_extractor import TimeExtractor
# from tower_tracker import TowerTracker

 
FOLDER = "C:/MyStuff/ClashRoyal/game_frames"
step = 35

def main():
    global step
    
    window_capture = WindowCapture("LDPlayer").start()
    # card_matcher = CardMatcher()
    # elixir_reader = ElixirExtractor()
    # timer_reader = TimeExtractor()
    # tower_tracker = TowerTracker()
    troop_detector = TroopDetector()
    board = BoardState()
    last_print = 0.0

    hand_res = {}

    for frame in window_capture.frames(fps=30): 

        step += 1
        static_elements = extract_static_elements(frame)
        # cv2.imwrite(f"{FOLDER}/frame_{step}.png", frame)
                
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
      
        # YOLO troop tracking: each troop keeps the same ID across frames
        tracks, annotated_frame = troop_detector.track(frame)
        board.update(tracks, frame)

        # Draw team + tile under each troop (red = enemy, blue = user, grey = team still being voted)
        for troop in board.snapshot():
            x1, y1, x2, y2 = troop["box"]
            if not troop["team_decided"]:
                colour = (160, 160, 160)
            else:
                colour = (0, 0, 255) if troop["team"] == "enemy" else (255, 128, 0)
            label = f"{troop['team']} {troop['tile']}" + ("" if troop["visible"] else " (held)")
            cv2.putText(annotated_frame, label, (x1, y2 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3)
            cv2.putText(annotated_frame, label, (x1, y2 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)
        cv2.imshow("Troop Detection", annotated_frame)

        # Show the battlefield once a second, replacing the previous reading (clear, then print)
        if time.monotonic() - last_print >= 1.0:
            last_print = time.monotonic()
            os.system("cls" if os.name == "nt" else "clear")
            print("\n".join(board.describe()) or "(no troops)")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            window_capture.stop()
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()



 
