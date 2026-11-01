# Original bot pipeline is preserved below as comments while testing YOLO.

import argparse
from pathlib import Path
import time

import cv2

from window_capture import WindowCapture
from yolo_preview import YoloPreview


# Use "1" or "2" for one model, or "both" for the combined preview.
MODEL_NUMBER = "both"
CONFIDENCE = 0.25
IMGSZ = 640
WINDOW_TITLE = "LDPlayer"
# The hand starts at about 80% of the captured image height. These models
# were trained on the battlefield; keep card artwork out of detection.
ARENA_BOTTOM = 0.80
MODEL_FOLDER = Path(__file__).resolve().parents[1] / "tempYOLO"


def main():
    parser = argparse.ArgumentParser(description="Live preview of one or both KataCR YOLO models.")
    parser.add_argument("--model", choices=("1", "2", "both"), default=MODEL_NUMBER)
    parser.add_argument("--conf", type=float, default=CONFIDENCE)
    parser.add_argument("--imgsz", type=int, default=IMGSZ)
    parser.add_argument("--window", default=WINDOW_TITLE, help="Emulator window title")
    args = parser.parse_args()
    if not 0 < args.conf <= 1:
        parser.error("--conf must be greater than 0 and at most 1")
    if args.imgsz <= 0:
        parser.error("--imgsz must be positive")

    model_numbers = (1, 2) if args.model == "both" else (int(args.model),)
    model_paths = [MODEL_FOLDER / f"detector{number}_v0.7.13.pt" for number in model_numbers]
    for model_path in model_paths:
        if not model_path.is_file():
            parser.error(f"Model file not found: {model_path}")
    detector = YoloPreview(model_paths, confidence=args.conf, imgsz=args.imgsz)
    print(f"Models: {', '.join(path.name for path in model_paths)} | device: {detector.device} | confidence: {args.conf}")
    print("Play a game in LDPlayer. Focus the preview and press Q or Esc to quit.")
    preview_title = "YOLO models 1 + 2" if args.model == "both" else f"YOLO model {args.model}"
    capture = WindowCapture(args.window)
    try:
        capture.start()
        cv2.namedWindow(preview_title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(preview_title, 450, 800)
        for frame in capture.frames(fps=10):
            started = time.perf_counter()
            arena_bottom = max(1, round(frame.shape[0] * ARENA_BOTTOM))
            arena, count = detector.detect(frame[:arena_bottom])
            annotated = frame.copy()
            annotated[:arena_bottom] = arena
            fps = 1.0 / max(time.perf_counter() - started, 1e-6)
            model_label = "Models 1+2" if args.model == "both" else f"Model {args.model}"
            label = f"{model_label} | {count} boxes | {fps:.1f} FPS | Q/Esc: quit"
            cv2.putText(annotated, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
            cv2.putText(annotated, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            cv2.imshow(preview_title, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if cv2.getWindowProperty(preview_title, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    except RuntimeError as error:
        parser.exit(1, f"{error}\nCheck that LDPlayer is running, or use --window with its title.\n")
    finally:
        capture.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


# ================= ORIGINAL MAIN (COMMENTED OUT) =================
# from turtle import width
# import numpy as np
# import cv2
# # from ultralytics import YOLO
#
# import os
# import time
#
# from troop_detector import TroopDetector
# from game_state import GameState
# from rule_based_brain import RuleBasedBrain
# from llm_brain import LLMBrain
# from emulator_input import EmulatorInput
#
#
# from window_capture import WindowCapture
# from static_elements_extractor import extract_static_elements
# # from card_matching import CardMatcher
# # from elixir_detection_ocr import ElixirExtractor
# # from timer_extractor import TimeExtractor
# # from tower_tracker import TowerTracker
#
#
# FOLDER = "C:/MyStuff/ClashRoyal/game_frames"
# step = 35
#
# DECISION_INTERVAL = 0.5     # seconds between bot decisions
# USE_LLM_BRAIN = False       # True = Groq LLM decides (needs GROQ_API_KEY), False = rule-based
#
# def main():
#     global step
#
#     window_capture = WindowCapture("LDPlayer").start()
#     # card_matcher = CardMatcher()
#     # elixir_reader = ElixirExtractor()
#     # timer_reader = TimeExtractor()
#     # tower_tracker = TowerTracker()
#     troop_detector = TroopDetector()
#     game_state = GameState()       # troops + hand + elixir + timer + towers, all in one place
#     brain = LLMBrain() if USE_LLM_BRAIN else RuleBasedBrain()
#     emulator = EmulatorInput()
#     last_print = 0.0
#
#     # Bot starts PAUSED: press "p" in the window to start/stop it playing cards
#     bot_playing = False
#     last_decision = 0.0
#     last_action = "none yet"
#
#     hand_res = {}
#
#     for frame in window_capture.frames(fps=30):
#
#         step += 1
#         static_elements = extract_static_elements(frame)
#         # cv2.imwrite(f"{FOLDER}/frame_{step}.png", frame)
#
#         for name, element in static_elements.items():
#             pass
#
#
#
#             # cv2.imwrite(f"./temp/debug_{name}.png", element)
#
#             # if name.startswith("hand"):
#             #     card_name, score = card_matcher.identify(element)
#             #     match_name = card_name if card_name is not None else "NO MATCH"
#             #     hand_res[name] = match_name
#             #     print(f"{name}: {match_name} (score: {score:.4f})")
#
#             # if name.startswith("elixir"):
#             #     elixir_value = elixir_reader.read(element, debug=False)
#             #     print(f"{name}: {elixir_value}")
#
#             # if name.endswith("king"):
#             #     tower_hp = tower_tracker.update(name, element)
#             #     print(f"{name}: {tower_hp}")
#             #     print(f"{name}: {tower_tracker.is_king_active(name)}")
#             #     cv2.imwrite(f"./temp/debug_{name}.png", element)
#
#                 # if (tower_tracker.is_dead(name)):
#                 #     print(f"{name}: DEAD")
#                 # else:
#                 #     print(f"{name}: {tower_hp}")
#
#             #     cv2.imwrite(f"./temp/debug_{name}.png", element)
#             #     step += 1
#
#             # if (name.startswith("timer")):
#                 # cv2.imwrite(f"./temp/debug_{name}.png", element)
#                 # timer_value = timer_reader.read(element)
#                 # print(f"{timer_value}")
#                 # cv2.imwrite(f"./temp/debug_{name}.png", element)
#                 # step += 1
#
#
#
#         # cv2.imshow("AI Vision", frame)
#
#         # YOLO troop tracking: each troop keeps the same ID across frames
#         tracks, annotated_frame = troop_detector.track(frame)
#         game_state.update(frame, tracks)
#
#         # Draw team + tile under each troop (red = enemy, blue = user, grey = team still being voted)
#         for troop in game_state.board.snapshot():
#             x1, y1, x2, y2 = troop["box"]
#             if not troop["team_decided"]:
#                 colour = (160, 160, 160)
#             else:
#                 colour = (0, 0, 255) if troop["team"] == "enemy" else (255, 128, 0)
#             label = f"{troop['team']} {troop['tile']}" + ("" if troop["visible"] else " (held)")
#             cv2.putText(annotated_frame, label, (x1, y2 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3)
#             cv2.putText(annotated_frame, label, (x1, y2 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)
#         status = "BOT PLAYING (p to pause)" if bot_playing else "BOT PAUSED (p to play)"
#         cv2.putText(annotated_frame, status, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
#         cv2.putText(annotated_frame, status, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#                     (0, 255, 0) if bot_playing else (0, 255, 255), 1)
#         cv2.imshow("Troop Detection", annotated_frame)
#
#         # Bot: decide + play every DECISION_INTERVAL, not every frame
#         if bot_playing and time.monotonic() - last_decision >= DECISION_INTERVAL:
#             last_decision = time.monotonic()
#             state = game_state.snapshot()
#             if USE_LLM_BRAIN:
#                 action = brain.decide(game_state)     # non-blocking; None while the model is thinking
#             else:
#                 action = brain.decide(state["troops"], state["hand"], state["elixir"])
#             if action is not None:
#                 emulator.play_card(action["slot"], *action["tile"])
#                 game_state.match.record_our_play(action["card"], action["tile"], time.monotonic())
#                 last_action = f"{action['card']} -> {action['tile']} ({action['reason']})"
#
#         # Show the battlefield once a second, replacing the previous reading (clear, then print)
#         if time.monotonic() - last_print >= 1.0:
#             last_print = time.monotonic()
#             os.system("cls" if os.name == "nt" else "clear")
#             # Only the match tracker (enemy elixir / cards / plays) for now -- the rest is commented out
#             # print(status)
#             # print(f"last action: {last_action}")
#             # if USE_LLM_BRAIN:
#             #     last_took = "-" if brain.last_request_seconds is None else f"{brain.last_request_seconds:.2f}s"
#             #     print(f"llm requests: {brain.request_count} | last request took: {last_took}")
#             #     print(f"llm reply: {brain.last_reply}")
#             #     if brain.last_error:
#             #         print(f"llm ERROR: {brain.last_error}")
#             # print("---")
#             # print("\n".join(game_state.describe()))
#             # lines = game_state.describe()
#             # print("\n".join(lines[:lines.index("troops:")]))   # troop list hidden for now
#             print("\n".join(game_state.match.describe()))
#
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord("p"):
#             bot_playing = not bot_playing
#         if key == ord("q"):
#             window_capture.stop()
#             break
#
#     cv2.destroyAllWindows()
#
#
# if __name__ == "__main__":
#     main()
#
#
#
#
