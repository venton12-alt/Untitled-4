"""Zeichnen des Overlays (FOV-Kreis + ESP-Boxen) auf das Videobild.

Das Overlay wird in reinen Visualisierungs-Layern gezeichnet und bei
Transparenz < 1.0 weich mit dem Videobild vermischt.
"""

import cv2

from ..config import OverlaySettings
from ..detection.detector import Detection


def draw_overlay(frame, detections: list[Detection], settings: OverlaySettings):
    """Gibt ein neues BGR-Frame mit eingezeichnetem Overlay zurück."""
    out = frame

    if settings.fov_enabled:
        height, width = frame.shape[:2]
        layer = out.copy()
        cv2.circle(
            layer,
            (width // 2, height // 2),
            max(4, settings.fov_radius),
            settings.fov_color,
            settings.fov_thickness,
            cv2.LINE_AA,
        )
        out = _blend(out, layer, settings.fov_opacity)

    if settings.esp_enabled:
        layer = out.copy()
        for det in detections:
            x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
            cv2.rectangle(layer, (x1, y1), (x2, y2), settings.esp_color,
                          settings.esp_thickness, cv2.LINE_AA)
            if settings.show_confidence:
                label = f"{det.conf:.0%}"
                pos = (x1, y1 - 6 if y1 - 6 > 14 else y1 + 16)
                cv2.putText(layer, label, pos, cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, settings.esp_color, 1, cv2.LINE_AA)
        out = _blend(out, layer, settings.esp_opacity)

    return out


def _blend(base, layer, alpha: float):
    if alpha >= 0.995:
        return layer
    return cv2.addWeighted(layer, alpha, base, 1.0 - alpha, 0)
