import threading
import time

import cv2
from windows_capture import WindowsCapture, Frame, InternalCaptureControl

# Crop pixels that are outside the useful game area on each captured frame.
CROP_TOP_PX = 40
CROP_RIGHT_PX = 40


class WindowCapture:
    """
    Captures a window via the Windows Graphics Capture API in the background.
    Conversion (BGRA -> BGR) only happens when a frame is actually pulled,
    so unused frames cost nothing.
    """

    def __init__(self, window_title: str):
        self.window_title = window_title
        self._latest_bgra = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._capture_error = None

        # This is a blocking call, so we need to use threads
        self._capture = WindowsCapture(
            cursor_capture=False,
            draw_border=False,
            monitor_index=None,
            window_name=window_title,
        )

        @self._capture.event
        def on_frame_arrived(frame: Frame, capture_control: InternalCaptureControl):

            # Cropping out the top and right borders of the emulator window
            cropped_bgra = frame.frame_buffer[CROP_TOP_PX:, :-CROP_RIGHT_PX]

            with self._lock:
                # Keep the frame in BGRA; convert to BGR only when it is requested.
                self._latest_bgra = cropped_bgra
            if self._stop_event.is_set():
                capture_control.stop()

        @self._capture.event
        def on_closed():
            self._stop_event.set()

        self._thread = threading.Thread(target=self._run_capture, daemon=True)

    def _run_capture(self):
        try:
            self._capture.start()
        except Exception as error:
            # Forward background failures to the caller instead of leaving it
            # waiting forever for a frame from a missing emulator window.
            self._capture_error = error
            self._stop_event.set()

    def _check_capture_error(self):
        if self._capture_error is not None:
            raise RuntimeError(
                f"Could not capture window {self.window_title!r}: {self._capture_error}"
            ) from self._capture_error

    def start(self):
        """Start background capture. Non-blocking."""
        self._thread.start()
        return self

    def get_frame(self):
        """Return the latest frame as BGR (numpy array), or None if not ready yet."""
        self._check_capture_error()
        with self._lock:
            bgra = self._latest_bgra
        if bgra is None:
            return None
        return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)

    def frames(self, fps: float):
        """
        Generator: yields one frame at the requested rate.
        Caller (main.py) controls pacing.
        """
        interval = 1.0 / fps
        while not self._stop_event.is_set():
            t0 = time.perf_counter()

            frame = self.get_frame()
            if frame is not None:
                yield frame

            elapsed = time.perf_counter() - t0
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        self._check_capture_error()

    def stop(self):
        """Signal the capture session to close."""
        self._stop_event.set()
