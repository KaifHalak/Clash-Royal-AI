from pathlib import Path

import cv2


def to_ratio(x1, y1, x2, y2, w, h):
    return (x1 / w, y1 / h, x2 / w, y2 / h)


w, h = 900, 1600

REGIONS = {
    "elixir_bar": to_ratio(240, 1520, 295, 1575, w, h),
    "hand_1": to_ratio(203, 1331, 365, 1520, w, h),
    "hand_2": to_ratio(368, 1331, 535, 1520, w, h),
    "hand_3": to_ratio(533, 1331, 701, 1520, w, h),
    "hand_4": to_ratio(702, 1331, 878, 1520, w, h),
    "next_card": to_ratio(47, 1491, 120, 1578, w, h),
    "tower_user_left": to_ratio(182, 994, 250, 1020, w, h),
    "tower_user_right": to_ratio(652, 994, 708, 1020, w, h),
    "tower_enemy_left": to_ratio(180, 213, 250, 240, w, h),
    "tower_enemy_right": to_ratio(650, 213, 720, 240, w, h),
    "timer": to_ratio(750, 36, 900, 85, w, h),
}


def crop_region(frame, box):
    frame_h, frame_w = frame.shape[:2]
    x1, y1, x2, y2 = box
    return frame[
        int(y1 * frame_h) : int(y2 * frame_h),
        int(x1 * frame_w) : int(x2 * frame_w),
    ]


def load_local_image(image_path: str | Path):
    """Load a local image as an OpenCV BGR frame."""
    image_path = Path(image_path)
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return frame


def extract_static_elements(frame):
    """Return all configured static regions from a frame."""
    return {name: crop_region(frame, region) for name, region in REGIONS.items()}


if __name__ == "__main__":
    # Replace this path with any local image you want to inspect.
    frame = load_local_image("C:/MyStuff/ClashRoyal/img/img-1.png")
    elements = extract_static_elements(frame)

    # for name, element in elements.items():
    #     cv2.imwrite(f"{name}.png", element)
    #     print(f"Saved {name}.png")
