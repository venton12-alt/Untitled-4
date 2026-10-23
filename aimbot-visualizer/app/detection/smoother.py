"""Anti-Flackern: Boxen werden über kurzzeitige Erkennungslücken gehalten
und ihre Positionen leicht geglättet (IoU-Matching zwischen Frames)."""

import time

from .detector import Detection


def _iou(a: Detection, b: Detection) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class DetectionSmoother:
    """Führt Erkennungen über Frames hinweg zusammen.

    - Passende Boxen (IoU > Schwellwert) werden geglättet statt hart neu gesetzt.
    - Verlorene Boxen bleiben für persist_ms sichtbar (weniger Flackern).
    """

    MATCH_IOU = 0.3
    SMOOTH_ALPHA = 0.4   # Anteil der neuen Position (1.0 = keine Glättung)

    def __init__(self, persist_ms: int = 150):
        self.persist_ms = persist_ms
        self._tracks: list[dict] = []

    def update(self, detections: list[Detection]) -> list[Detection]:
        now = time.monotonic()
        for det in detections:
            best, best_iou = None, self.MATCH_IOU
            for track in self._tracks:
                iou = _iou(track["det"], det)
                if iou > best_iou:
                    best, best_iou = track, iou
            if best is not None:
                a = self.SMOOTH_ALPHA
                old = best["det"]
                best["det"] = Detection(
                    x1=a * det.x1 + (1 - a) * old.x1,
                    y1=a * det.y1 + (1 - a) * old.y1,
                    x2=a * det.x2 + (1 - a) * old.x2,
                    y2=a * det.y2 + (1 - a) * old.y2,
                    conf=det.conf,
                    cls=det.cls,
                )
                best["last_seen"] = now
            else:
                self._tracks.append({"det": det, "last_seen": now})

        # Abgelaufene Tracks entfernen
        self._tracks = [
            t for t in self._tracks if (now - t["last_seen"]) * 1000.0 <= self.persist_ms
        ]
        return [t["det"] for t in self._tracks]

    def reset(self) -> None:
        self._tracks.clear()
