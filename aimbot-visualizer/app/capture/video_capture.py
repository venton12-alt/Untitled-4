"""Erfassung von Videodateien und Webcams über OpenCV."""

import time

import cv2

from .base import CaptureSource


class VideoCapture(CaptureSource):
    """Liest Videodateien (optional in Schleife) oder Webcam-Streams."""

    def __init__(self, source, loop: bool = False, max_fps: int = 60):
        super().__init__()
        self.source = source          # Dateipfad (str) oder Kameraindex (int)
        self.loop = loop
        self.max_fps = max_fps
        self._cap = None
        self._last_read = 0.0

    def open(self) -> None:
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"Videoquelle konnte nicht geöffnet werden: {self.source}")
        if isinstance(self.source, int):
            # Puffer minimal halten -> geringere Latenz bei Webcams
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap = cap

    def read(self):
        cap = self._cap
        if cap is None:
            return None
        interval = 1.0 / self.max_fps
        wait = interval - (time.perf_counter() - self._last_read)
        if wait > 0:
            time.sleep(min(0.005, wait))
        self._last_read = time.perf_counter()
        ok, frame = cap.read()
        if not ok and self.loop:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        if not ok:
            return None
        self._tick()
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
