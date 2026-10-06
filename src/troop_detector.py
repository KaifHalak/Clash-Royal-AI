from ultralytics import YOLO

# ================= CONFIG =================
MODEL_PATH = "C:/MyStuff/ClashRoyal/best.pt"  
IMGSZ = 960            # img size the model was trained on
CONFIDENCE = 0.25      # boxes below this confidence are dropped

# Tracking (ByteTrack, built into Ultralytics)
TRACKER_CONFIG = "bytetrack.yaml"   # Ultralytics' default ByteTrack settings
TRACK_CONFIDENCE = 0.1              # lower than CONFIDENCE on purpose: ByteTrack uses weak boxes to keep tracks alive


class TroopDetector:
    def __init__(self, model_path=MODEL_PATH):
        # Loaded once 
        self.model = YOLO(model_path)

    def detect(self, frame):
        """Run YOLO on one BGR frame.
        Returns (detections, annotated_frame):
            detections: list of (class_name, confidence, (x1, y1, x2, y2))
            annotated_frame: copy of the frame with boxes, names and confidences drawn on it
        """
        result = self.model.predict(frame, imgsz=IMGSZ, conf=CONFIDENCE, verbose=False)[0]

        detections = []
        for box in result.boxes:
            class_name = result.names[int(box.cls)]
            confidence = float(box.conf)
            x1, y1, x2, y2 = (int(value) for value in box.xyxy[0])
            detections.append((class_name, confidence, (x1, y1, x2, y2)))

        return detections, result.plot()

    def track(self, frame):
        """Like detect(), but also follows each troop across frames and gives it an ID.
        Call it on every frame, in order -- persist=True makes it remember the previous frames.
        Returns (tracks, annotated_frame):
            tracks: list of (track_id, class_name, confidence, (x1, y1, x2, y2))
            annotated_frame: copy of the frame with boxes, IDs, names and confidences drawn on it
        """
        result = self.model.track(frame, imgsz=IMGSZ, conf=TRACK_CONFIDENCE, persist=True,
                                  tracker=TRACKER_CONFIG, verbose=False)[0]

        tracks = []
        if result.boxes.id is None:     # no tracked troops in this frame
            return tracks, result.plot()

        for box in result.boxes:
            track_id = int(box.id)
            class_name = result.names[int(box.cls)]
            confidence = float(box.conf)
            x1, y1, x2, y2 = (int(value) for value in box.xyxy[0])
            tracks.append((track_id, class_name, confidence, (x1, y1, x2, y2)))

        return tracks, result.plot()
