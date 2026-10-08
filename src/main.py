from turtle import width
import numpy as np
import cv2
# from ultralytics import YOLO

import os
import time

from troop_detector import TroopDetector
from game_state import GameState
from rule_based_brain import RuleBasedBrain
from emulator_input import EmulatorInput


from window_capture import WindowCapture
from static_elements_extractor import extract_static_elements
# from card_matching import CardMatcher
# from elixir_detection_ocr import ElixirExtractor
# from timer_extractor import TimeExtractor
# from tower_tracker import TowerTracker


FOLDER = "C:/MyStuff/ClashRoyal/game_frames"
step = 35

DECISION_INTERVAL = 0.5     # seconds between bot decisions

def main():
    global step

    window_capture = WindowCapture("LDPlayer").start()
    # card_matcher = CardMatcher()
    # elixir_reader = ElixirExtractor()
    # timer_reader = TimeExtractor()
    # tower_tracker = TowerTracker()
    troop_detector = TroopDetector()
    game_state = GameState()       # troops + hand + elixir + timer + towers, all in one place
    brain = RuleBasedBrain()
    emulator = EmulatorInput()
    last_print = 0.0

    # Bot starts PAUSED: press "p" in the window to start/stop it playing cards
    bot_playing = False
    last_decision = 0.0
    last_action = "none yet"

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
        game_state.update(frame, tracks)

        # Draw team + tile under each troop (red = enemy, blue = user, grey = team still being voted)
        for troop in game_state.board.snapshot():
            x1, y1, x2, y2 = troop["box"]
            if not troop["team_decided"]:
                colour = (160, 160, 160)
            else:
                colour = (0, 0, 255) if troop["team"] == "enemy" else (255, 128, 0)
            label = f"{troop['team']} {troop['tile']}" + ("" if troop["visible"] else " (held)")
            cv2.putText(annotated_frame, label, (x1, y2 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3)
            cv2.putText(annotated_frame, label, (x1, y2 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)
        status = "BOT PLAYING (p to pause)" if bot_playing else "BOT PAUSED (p to play)"
        cv2.putText(annotated_frame, status, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(annotated_frame, status, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0) if bot_playing else (0, 255, 255), 1)
        cv2.imshow("Troop Detection", annotated_frame)

        # Bot: decide + play every DECISION_INTERVAL, not every frame
        if bot_playing and time.monotonic() - last_decision >= DECISION_INTERVAL:
            last_decision = time.monotonic()
            state = game_state.snapshot()
            action = brain.decide(state["troops"], state["hand"], state["elixir"])
            if action is not None:
                emulator.play_card(action["slot"], *action["tile"])
                last_action = f"{action['card']} -> {action['tile']} ({action['reason']})"

        # Show the battlefield once a second, replacing the previous reading (clear, then print)
        if time.monotonic() - last_print >= 1.0:
            last_print = time.monotonic()
            os.system("cls" if os.name == "nt" else "clear")
            print(status)
            print(f"last action: {last_action}")
            print("---")
            print("\n".join(game_state.describe()))

        key = cv2.waitKey(1) & 0xFF
        if key == ord("p"):
            bot_playing = not bot_playing
        if key == ord("q"):
            window_capture.stop()
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()



 
