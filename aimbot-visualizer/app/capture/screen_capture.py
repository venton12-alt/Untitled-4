"""Echtzeit-Bildschirmaufnahme mit mss (schnell, Windows/macOS/Linux)."""

import time

import mss
import numpy as np

from .base import CaptureSource


class ScreenCapture(CaptureSource):
    """Erfasst einen Monitor oder einen Bildschirmbereich in Echtzeit."""

    name = "Bildschirm"

    def __init__(self, monitor_index: int = 1, region: tuple | None = None, max_fps: int = 60):
        super().__init__()
        self.monitor_index = monitor_index
        self.region = region          # (left, top, width, height) relativ zum Monitor
        self.max_fps = max_fps
        self._sct = None
        self._last_read = 0.0

    def open(self) -> None:
        self._sct = mss.mss()
        monitors = self._sct.monitors
        if self.monitor_index >= len(monitors):
            raise RuntimeError(
                f"Monitor {self.monitor_index} nicht gefunden "
                f"(verfügbar: 0..{len(monitors) - 1})."
            )
        monitor = monitors[self.monitor_index]
        if self.region:
            left, top, width, height = self.region
            self._region = {
                "left": monitor["left"] + left,
                "top": monitor["top"] + top,
                "width": width,
                "height": height,
            }
        else:
            self._region = monitor

    def read(self):
        if self._sct is None:
            return None
        # Ratenbegrenzung: schont CPU, ohne spürbare Latenz zu erzeugen
        interval = 1.0 / self.max_fps
        wait = interval - (time.perf_counter() - self._last_read)
        if wait > 0:
            time.sleep(min(0.005, wait))
        self._last_read = time.perf_counter()
        try:
            shot = self._sct.grab(self._region)
        except Exception:
            return None
        # BGRA -> BGR, fortlaufender Speicher für OpenCV/NumPy
        frame = np.ascontiguousarray(np.asarray(shot)[:, :, :3])
        self._tick()
        return frame

    def close(self) -> None:
        if self._sct is not None:
            self._sct.close()
            self._sct = None
