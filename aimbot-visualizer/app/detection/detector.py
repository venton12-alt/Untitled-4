"""KI-Objekterkennung mit Ultralytics YOLO.

Lädt das Modell erst beim ersten Detect-Aufruf (lazily), damit die
Oberfläche sofort startet, auch wenn der Modell-Download dauert.
"""

from dataclasses import dataclass

import numpy as np

from ..config import DetectionSettings


@dataclass
class Detection:
    """Eine erkannte Objektbox in Bildkoordinaten (Pixel)."""

    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls: int

    @property
    def box(self) -> tuple:
        return (self.x1, self.y1, self.x2, self.y2)


class Detector:
    """Führt die YOLO-Inferenz aus und filtert nach Zielklassen."""

    def __init__(self, settings: DetectionSettings):
        self.settings = settings
        self._model = None
        self.last_error: str | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO  # erst hier importieren (teures Paket)
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics ist nicht installiert. Bitte 'pip install -r requirements.txt' "
                "ausführen."
            ) from exc
        try:
            self._model = YOLO(self.settings.model_path)
        except Exception as exc:
            raise RuntimeError(
                f"Modell '{self.settings.model_path}' konnte nicht geladen werden "
                f"(beim ersten Start wird yolov8n.pt automatisch heruntergeladen): {exc}"
            ) from exc
        if self.settings.device:
            try:
                self._model.to(self.settings.device)
            except Exception as exc:
                raise RuntimeError(f"Gerät '{self.settings.device}' nicht nutzbar: {exc}") from exc

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        """Erkennt Objekte in einem BGR-Frame und liefert gefilterte Boxen."""
        self._ensure_model()
        s = self.settings
        classes = list(s.target_class_ids) if s.target_class_ids else None
        results = self._model.predict(
            frame_bgr,
            conf=s.conf_threshold,
            iou=s.iou_threshold,
            imgsz=s.imgsz,
            classes=classes,
            device=s.device or None,
            verbose=False,
        )
        detections: list[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes_arr = boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, classes_arr):
                detections.append(
                    Detection(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                              conf=float(conf), cls=int(cls))
                )
        return detections
