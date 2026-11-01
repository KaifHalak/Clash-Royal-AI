"""Inference-only support for the two downloaded KataCR v0.7.13 checkpoints.

Reference: https://github.com/wty-yy/KataCR/tree/master/katacr/yolov8
These use a regular YOLOv8 network, with a final output for team membership.
The custom checkpoint classes otherwise change training, not inference.
"""

import pickle
import types
from pathlib import Path

import numpy as np
import torch
from torchvision.ops import batched_nms
from ultralytics.data.augment import LetterBox
from ultralytics.engine.results import Results
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import ops
from ultralytics.utils.nms import non_max_suppression


# Exclude these labels from both model previews, including the box count.
DISCARDED_LABELS = {
    "king-tower",
    "queen-tower",
    "cannoneer-tower",
    "dagger-duchess-tower",
    "dagger-duchess-tower-bar",
    "tower-bar",
    "king-tower-bar",
    "bar",
    "bar-level",
    "clock",
    "emote",
    "elixir",
}


class _UnusedTrainingState:
    """Load saved loss/assigner state without importing KataCR's training stack."""


class _CheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "katacr.yolov8.custom_model":
            if name == "CRDetectionModel":
                return DetectionModel
            if name in {"CRDetectionLoss", "CRTaskAlignedAssigner"}:
                return _UnusedTrainingState
        return super().find_class(module, name)


class YoloPreview:
    def __init__(self, model_path, confidence=0.25, imgsz=896):
        model_paths = [model_path] if isinstance(model_path, (str, Path)) else list(model_path)
        if not model_paths:
            raise ValueError("At least one model path is required.")
        checkpoint_pickle = types.ModuleType("katacr_checkpoint_pickle")
        checkpoint_pickle.Unpickler = _CheckpointUnpickler
        checkpoint_pickle.load = pickle.load
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.models = []
        label_ids = {}
        stride = 1
        for path in model_paths:
            # These .pt files contain complete model objects, not just weights.
            checkpoint = torch.load(
                path, map_location="cpu", pickle_module=checkpoint_pickle,
                weights_only=False,
            )
            model = checkpoint.get("ema") or checkpoint["model"]
            if hasattr(model, "criterion"):
                del model.criterion  # Training is deliberately unsupported here.
            model = model.to(self.device).float().eval()
            class_count = model.model[-1].nc - 1
            if model.names[class_count] != "padding_belong":
                raise ValueError("Expected a KataCR checkpoint with a final team output.")
            classes = [
                idx for idx, name in model.names.items()
                if not name.startswith("padding_") and name not in DISCARDED_LABELS
            ]
            # Each checkpoint has its own class numbering. Merge by label name.
            class_map = torch.full((class_count,), -1, device=self.device, dtype=torch.long)
            for idx in classes:
                name = model.names[idx]
                class_map[idx] = label_ids.setdefault(name, len(label_ids))
            self.models.append((model, class_count, classes, class_map))
            stride = max(stride, int(model.stride.max()))
        self.names = {idx: name for name, idx in label_ids.items()}
        self.confidence = confidence
        imgsz = ((imgsz + stride - 1) // stride) * stride
        self.letterbox = LetterBox((imgsz, imgsz), auto=True, stride=stride)

    @torch.inference_mode()
    def detect(self, frame):
        image = self.letterbox(image=frame)
        # Capture produces BGR/HWC; the network expects RGB/CHW in [0, 1].
        image = np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1))
        tensor = torch.from_numpy(image).to(self.device).float().unsqueeze(0) / 255.0
        detections = []
        for model, class_count, classes, class_map in self.models:
            prediction = model(tensor)[0]
            # Treat the team output as extra data, never as a unit class.
            boxes = non_max_suppression(
                prediction, conf_thres=self.confidence, iou_thres=0.7,
                nc=class_count, classes=classes,
            )[0][:, :6]
            boxes[:, 5] = class_map[boxes[:, 5].long()]
            detections.append(boxes)
        boxes = torch.cat(detections)
        if len(boxes):
            # Remove duplicate detections of the same label across models.
            keep = batched_nms(boxes[:, :4], boxes[:, 4], boxes[:, 5].long(), 0.7)
            boxes = boxes[keep]
        boxes[:, :4] = ops.scale_boxes(tensor.shape[2:], boxes[:, :4], frame.shape)
        # Results interprets seven columns as tracking; use the six box columns
        # because this preview only shows current detections, without track IDs.
        result = Results(frame, path="", names=self.names, boxes=boxes.cpu())
        return result.plot(), len(boxes)
