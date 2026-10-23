"""Zentrale Konfiguration der Anwendung.

Alle einstellbaren Parameter werden über Dataclasses gebündelt und
zur Laufzeit von der Benutzeroberfläche verändert (Live-Update).
"""

from dataclasses import dataclass, field


@dataclass
class OverlaySettings:
    """Einstellungen für FOV-Kreis und ESP-Boxen."""

    # --- FOV-Kreis ---
    fov_enabled: bool = True
    fov_radius: int = 180          # Radius in Pixeln
    fov_color: tuple = (0, 255, 120)    # BGR (hellgrün)
    fov_thickness: int = 2
    fov_opacity: float = 1.0       # 0.0 (unsichtbar) .. 1.0 (deckend)

    # --- ESP-Boxen ---
    esp_enabled: bool = True
    esp_color: tuple = (60, 60, 255)    # BGR (rot)
    esp_thickness: int = 2
    esp_opacity: float = 1.0
    show_confidence: bool = True   # Konfidenzwert über der Box anzeigen


@dataclass
class DetectionSettings:
    """Einstellungen für die KI-Objekterkennung."""

    # Pfad zum YOLO-Modell. Standard: yolov8n (COCO, Klasse 0 = "person").
    # Für spiel-spezifische Modelle (z. B. auf Valorant-Screenshots trainiert)
    # hier den Pfad zu einer eigenen .pt-Datei eintragen.
    model_path: str = "yolov8n.pt"

    conf_threshold: float = 0.45   # Erkennungsschwelle
    iou_threshold: float = 0.45    # NMS-IoU
    imgsz: int = 480               # kleinere Eingangsgröße = weniger Latenz
    target_class_ids: tuple = (0,) # COCO: 0 = "person"; bei eigenen Modellen anpassen
    device: str = ""               # "" = automatisch (CUDA falls verfügbar), sonst "cpu"

    # Anti-Flackern: Boxen bleiben bis zu so vielen Millisekunden sichtbar,
    # wenn die Erkennung sie kurzzeitig verliert.
    persist_ms: int = 150


@dataclass
class CaptureSettings:
    """Einstellungen für die Bildschirmaufnahme."""

    monitor_index: int = 1         # 1 = Hauptbildschirm (mss-Nummerierung)
    region: tuple | None = None    # (left, top, width, height) relativ zum Monitor, None = voll
    max_fps: int = 60              # Ratenbegrenzung der Erfassung
    loop_video: bool = True        # Videodateien in Schleife abspielen
