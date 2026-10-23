"""Thread-sichere Ablage für Laufzeit-Statistiken und Statusmeldungen.

Die Worker-Threads schreiben, die Oberfläche liest – im Gegensatz zum
Frame-Speicher sind hier alle Werte klein, ein Lock genügt.
"""

import threading


class Stats:
    def __init__(self):
        self._lock = threading.Lock()
        self.capture_fps = 0.0
        self.detect_fps = 0.0
        self.infer_ms = 0.0
        self.detection_count = 0
        self.status = "Bereit"
        self.error: str | None = None

    def set(self, **kwargs) -> None:
        with self._lock:
            for key, value in kwargs.items():
                setattr(self, key, value)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "capture_fps": self.capture_fps,
                "detect_fps": self.detect_fps,
                "infer_ms": self.infer_ms,
                "detection_count": self.detection_count,
                "status": self.status,
                "error": self.error,
            }

    def reset_runtime(self) -> None:
        self.set(capture_fps=0.0, detect_fps=0.0, infer_ms=0.0,
                 detection_count=0, error=None)
