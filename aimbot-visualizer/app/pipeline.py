"""Echtzeit-Pipeline: Erfassung -> KI-Erkennung -> Anzeige.

- FrameStore hält immer nur das jüngste Frame (Frames fallenlassen statt
  anstauen = geringe Latenz).
- CaptureWorker füllt den FrameStore, DetectionWorker rechnet YOLO und
  schreibt die geglätteten Erkennungen in einen DetectionStore.
- Beide Worker werden über ein Stop-Event sauber beendet.
"""

import threading
import time

import numpy as np

from .detection.detector import Detector
from .detection.smoother import DetectionSmoother
from .stats import Stats


class FrameStore:
    """Thread-sicherer Speicher für das aktuellste Frame (double-buffered)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None

    def put(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame

    def get(self) -> np.ndarray | None:
        with self._lock:
            return self._frame

    def clear(self) -> None:
        with self._lock:
            self._frame = None


class DetectionStore:
    """Thread-sicherer Speicher für die aktuell sichtbaren Erkennungen."""

    def __init__(self):
        self._lock = threading.Lock()
        self._detections: list = []

    def set(self, detections: list) -> None:
        with self._lock:
            self._detections = detections

    def get(self) -> list:
        with self._lock:
            return list(self._detections)


class CaptureWorker(threading.Thread):
    """Liest fortlaufend Frames aus der Quelle in den FrameStore."""

    def __init__(self, source, frame_store: FrameStore, stats: Stats,
                 stop_event: threading.Event):
        super().__init__(name="capture-worker", daemon=True)
        self.source = source
        self.frame_store = frame_store
        self.stats = stats
        self.stop_event = stop_event

    def run(self) -> None:
        consecutive_failures = 0
        try:
            self.source.open()
            self.stats.set(status=f"Erfassung aktiv ({self.source.name})")
        except Exception as exc:
            self.stats.set(status="Erfassung fehlgeschlagen", error=str(exc))
            return

        try:
            while not self.stop_event.is_set():
                frame = self.source.read()
                if frame is None:
                    consecutive_failures += 1
                    if consecutive_failures > 100:
                        self.stats.set(status="Quelle liefert keine Bilder",
                                       error="Erfassung endete (Quelle leer/gesperrt).")
                        break
                    time.sleep(0.01)
                    continue
                consecutive_failures = 0
                self.frame_store.put(frame)
                self.stats.set(capture_fps=self.source.fps)
        finally:
            self.source.close()
            self.frame_store.clear()


class DetectionWorker(threading.Thread):
    """Führt die YOLO-Erkennung auf dem jüngsten Frame aus."""

    def __init__(self, detector: Detector, frame_store: FrameStore,
                 detection_store: DetectionStore, stats: Stats,
                 stop_event: threading.Event):
        super().__init__(name="detection-worker", daemon=True)
        self.detector = detector
        self.frame_store = frame_store
        self.detection_store = detection_store
        self.stats = stats
        self.stop_event = stop_event
        self.smoother = DetectionSmoother(persist_ms=150)

    def run(self) -> None:
        detect_times: list[float] = []
        last_detect = 0.0
        while not self.stop_event.is_set():
            frame = self.frame_store.get()
            if frame is None:
                time.sleep(0.01)
                continue
            try:
                start = time.perf_counter()
                detections = self.detector.detect(frame)
                infer_ms = (time.perf_counter() - start) * 1000.0
            except Exception as exc:
                self.stats.set(status="Erkennung fehlgeschlagen", error=str(exc))
                return
            smoothed = self.smoother.update(detections)
            self.detection_store.set(smoothed)

            now = time.perf_counter()
            detect_times.append(now)
            while detect_times and now - detect_times[0] > 1.0:
                detect_times.pop(0)
            self.stats.set(
                detect_fps=len(detect_times) / max(1e-6, now - detect_times[0]),
                infer_ms=infer_ms,
                detection_count=len(smoothed),
            )
            last_detect = now
            # Ratenbegrenzung der KI: nötigenfalls kurze Pause, um CPU zu sparen
            if now - last_detect < 0.002:
                time.sleep(0.002)

    def stop(self) -> None:
        self.smoother.reset()


class Pipeline:
    """Verwaltet Start/Stopp der Worker für eine Erfassungsquelle."""

    def __init__(self, detector: Detector, stats: Stats):
        self.detector = detector
        self.stats = stats
        self.frame_store = FrameStore()
        self.detection_store = DetectionStore()
        self._stop_event = threading.Event()
        self._workers: list[threading.Thread] = []

    @property
    def running(self) -> bool:
        return any(w.is_alive() for w in self._workers)

    def start(self, source) -> None:
        if self.running:
            raise RuntimeError("Pipeline läuft bereits.")
        self._stop_event.clear()
        self.stats.reset_runtime()
        self.detector.last_error = None
        self.detection_store.set([])
        capture = CaptureWorker(source, self.frame_store, self.stats, self._stop_event)
        detection = DetectionWorker(self.detector, self.frame_store,
                                    self.detection_store, self.stats, self._stop_event)
        self._workers = [capture, detection]
        capture.start()
        detection.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop_event.set()
        for worker in self._workers:
            if worker.is_alive():
                worker.join(timeout=timeout)
        self._workers = []
        self.frame_store.clear()
        self.detection_store.set([])
        self.stats.set(status="Bereit")

    def shutdown(self) -> None:
        """Sauberes Beenden beim Schließen des Fensters."""
        self._stop_event.set()
        for worker in self._workers:
            worker.join(timeout=1.0)
        for worker in self._workers:
            if worker.is_alive():
                raise RuntimeError("Worker konnten nicht sauber beendet werden.")
