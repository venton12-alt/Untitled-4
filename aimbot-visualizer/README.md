# KI-Aimbot-Visualisierer (reine Visualisierung)

Dieses Programm simuliert die **Optik** eines KI-Aimbots für Aufnahmen
(YouTube, Demos): Es erfasst ein Gaming-Video (Bildschirm, Videodatei oder
Webcam) in Echtzeit, erkennt Personen mit einem YOLO-Modell und zeichnet
ESP-Boxen sowie einen zentralen FOV-Kreis über das Bild.

> ⚠️ **Wichtig:** Das Tool steuert **nicht** die Maus oder Tastatur, schreibt
> **keine** Eingaben, greift **nicht** in Spiele oder Prozesse ein und
> automatisiert **nichts**. Es ist ausschließlich ein Betrachter/Overlayer
> für Videomaterial – gedacht für Demonstrationen und Content-Erstellung.
> Bitte beachte die Nutzungsbedingungen der jeweiligen Spiele und Plattformen.

## Funktionen

- Echtzeit-Erfassung: Bildschirm (mss), Videodatei (MP4/MKV/AVI/…, mit Schleife) oder Webcam
- Start/Stopp der Erfassung per Button
- FOV-Kreis zentriert im Bild: Größe, Linienstärke, Transparenz einstellbar
- ESP-Boxen um erkannte Objekte: dynamisch in Position/Größe, mit Konfidenzanzeige
- YOLO-Erkennung (ultralytics), einstellbare Erkennungsschwelle, eigenes Modell ladbar
- Anti-Flackern: Boxen werden über kurze Erkennungslücken geglättet/gehalten
- Live-Anzeige: Erfassungs-FPS, Erkennungs-FPS, Inferenz-Zeit, Anzahl Erkennungen
- Sauberes Beenden: Threads werden gestoppt, Quellen geschlossen

## Installation

Python ≥ 3.10 wird benötigt (tkinter ist bei den meisten Distributionen
enthalten; unter Linux ggf. `sudo apt install python3-tk`).

```bash
cd aimbot-visualizer
python -m venv .venv

# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

Beim ersten Start lädt Ultralytics automatisch das Standardmodell
`yolov8n.pt` (~6 MB, Internetverbindung nötig). Es erkennt aus COCO die
Klasse **"person"** – in Spiel-Footage funktioniert das je nach Szene
unterschiedlich gut.

### Eigenes (spiel-spezifisches) Modell nutzen

Für stabilere Erkennung in Valorant-Material kann ein eigenes YOLO-Modell
verwendet werden (z. B. selbst auf Spiel-Screenshots trainiert oder ein
öffentlich verfügbares Community-Modell). In der Oberfläche über
**„Anderes Modell laden…“** auswählen, oder in `app/config.py` unter
`DetectionSettings.model_path` eintragen. Für eigene Klassen die
`target_class_ids` entsprechend anpassen (`(0,)` = erste Klasse des Modells).

## Ausführung

```bash
python main.py
```

Bedienung:

1. Quelle wählen (Bildschirm / Videodatei / Webcam; bei Datei: auswählen)
2. **Start** drücken – das Vorschau-Panel zeigt das Video mit Overlay
3. FOV/ESP ein-/ausschalten und Regler nach Wunsch einstellen (Live-Effekt)
4. Für Aufnahmen: das Vorschau-Fenster mit OBS o. Ä. aufnehmen – oder das
   Tool einfach neben/nebenüber zum Videoplayer laufen lassen und den
   gewünschten Bereich aufnehmen

## Architektur

```
aimbot-visualizer/
├── main.py                     # Einstiegspunkt
├── requirements.txt
└── app/
    ├── config.py               # Alle Einstellungen (Overlay, Erkennung, Erfassung)
    ├── stats.py                # Thread-sichere Laufzeit-Statistiken
    ├── pipeline.py             # Worker-Threads: Erfassung → Erkennung → Anzeige
    ├── capture/
    │   ├── base.py             # Schnittstelle aller Quellen (+FPS-Messung)
    │   ├── screen_capture.py   # Bildschirmaufnahme (mss)
    │   └── video_capture.py    # Videodatei/Webcam (OpenCV)
    ├── detection/
    │   ├── detector.py         # YOLO-Inferenz, Klassenfilter
    │   └── smoother.py         # Anti-Flackern (IoU-Matching, Persistenz)
    ├── overlay/
    │   └── renderer.py         # FOV-Kreis + ESP-Boxen zeichnen
    └── ui/
        └── main_window.py      # tkinter-Oberfläche (Haupt-Thread)
```

Datenfluss: **Erfassungs-Thread** schreibt immer nur das jüngste Frame in
einen FrameStore (alte Frames werden verworfen → geringe Latenz). Der
**Erkennungs-Thread** nimmt sich das jüngste Frame, rechnet YOLO, glättet
die Boxen und legt sie ab. Der **UI-Thread** zeichnet ~60×/s Frame + Overlay
und aktualisiert FPS-/Statusanzeigen. Threads laufen getrennt, damit die
Oberfläche flüssig bleibt, selbst wenn die Inferenz länger dauert.

## Tests

Ein Smoke-Test prüft Glättung, Overlay-Zeichnung und die komplette
Pipeline (mit synthetischem Video und Fake-Detector, ohne ML-Download):

```bash
PYTHONPATH=. python tests/smoke_test.py
```

## Performance-Tipps

- GPU nutzen: `pip install torch --index-url https://download.pytorch.org/whl/cu121`
  (ultralytics nutzt CUDA automatisch, sobald verfügbar) – sinkt die
  Inferenzzeit oft auf wenige Millisekunden.
- `imgsz` in `DetectionSettings` kleiner wählen (z. B. 320) für schnellere
  Erkennung auf CPUs.
- `max_fps` der Erfassung reduzieren, wenn die CPU ausgelastet ist.
- nur einen Bildschirmbereich (`CaptureSettings.region`) erfassen statt des
  ganzen Monitors.

## Fehlerbehebung

| Problem | Lösung |
| --- | --- |
| „Monitor nicht gefunden“ | `monitor_index` in `CaptureSettings` anpassen (mss: 0 = alle, 1 = Hauptmonitor) |
| Videodatei öffnet nicht | Codec fehlt; Datei vorher z. B. mit HandBrake zu H.264-MP4 konvertieren |
| Hohe CPU-Last | `imgsz` verringern, `max_fps` senken, GPU installieren |
| Modelldownload schlägt fehl | manuell von <https://github.com/ultralytics/assets/releases> laden und per „Anderes Modell laden…“ wählen |
| tkinter fehlt (Linux) | `sudo apt install python3-tk` |

## Rechtliches / Fair-Use-Hinweis

Dieses Werkzeug verändert kein Spiel, umgeht keine Anti-Cheat-Systeme und
automatisiert nichts – es analysiert nur Videobilder und zeichnet darauf.
Verwende es nur mit Material, an dem du die Rechte hast, und kennzeichne
simulierte Inhalte in Veröffentlichungen entsprechend.
