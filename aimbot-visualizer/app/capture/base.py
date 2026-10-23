"""Basisklasse für alle Videoquellen (Bildschirm, Datei, Webcam)."""

import time
from abc import ABC, abstractmethod


class CaptureSource(ABC):
    """Gemeinsame Schnittstelle aller Erfassungsquellen.

    Eine Quelle liefert über read() fortlaufend BGR-Frames (numpy-Arrays
    im OpenCV-Format) und zählt dabei ihre eigene Framerate.
    """

    name = "Basis"

    def __init__(self):
        self._frame_count = 0
        self._fps = 0.0
        self._last_t = time.perf_counter()

    @abstractmethod
    def open(self) -> None:
        """Ressourcen anfordern. Löst bei Fehlern eine Exception aus."""

    @abstractmethod
    def read(self):
        """Nächstes BGR-Frame liefern oder None (Quelle vorübergehend leer)."""

    @abstractmethod
    def close(self) -> None:
        """Ressourcen sauber freigeben."""

    @property
    def fps(self) -> float:
        return self._fps

    def _tick(self) -> None:
        """Framerate-Messung nach jedem gelesenen Frame aufrufen."""
        self._frame_count += 1
        now = time.perf_counter()
        elapsed = now - self._last_t
        if elapsed >= 1.0:
            self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._last_t = now
