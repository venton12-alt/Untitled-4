"""Smoke-Test der Pipeline ohne GUI und ohne YOLO-Download.

Erzeugt ein synthetisches Testvideo, betreibt Erfassungs- und
Erkennungs-Worker mit einem Fake-Detector und prüft Frame-Austausch,
Glättung/Persistenz, Overlay-Zeichnung und sauberes Stoppen.

Ausführen:  PYTHONPATH=. python tests/smoke_test.py
"""

import sys
import time

import cv2
import numpy as np

from app.config import DetectionSettings, OverlaySettings
from app.detection.detector import Detection
from app.detection.smoother import DetectionSmoother
from app.overlay import draw_overlay
from app.pipeline import Pipeline
from app.stats import Stats


def make_test_video(path: str, frames: int = 60, fps: int = 30) -> str:
    """Schreibt ein Video mit wanderndem hellem Rechteck (Pseudo-Gegner)."""
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (640, 360))
    for i in range(frames):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        x = 50 + int(i * 6)
        cv2.rectangle(frame, (x, 150), (x + 60, 260), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return path


class FakeDetector:
    """Erkennt das helle Rechteck deterministisch (kein ML nötig)."""

    def detect(self, frame: np.ndarray) -> list[Detection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = gray > 200
        ys, xs = np.nonzero(mask)
        if len(xs) < 50:
            return []
        return [Detection(x1=float(xs.min()), y1=float(ys.min()),
                          x2=float(xs.max()), y2=float(ys.max()),
                          conf=0.91, cls=0)]


def approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) < tol


def main() -> int:
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"[{'OK ' if ok else 'FEHL'}] {name}" + (f" – {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    # --- 1) Glättung/Persistenz ---
    smoother = DetectionSmoother(persist_ms=150)
    det = Detection(10, 10, 50, 50, 0.9, 0)
    out1 = smoother.update([det])
    check("Smoother: Box nach erstem Update sichtbar", len(out1) == 1)
    out2 = smoother.update([])  # Frame ohne Erkennung -> soll gehalten werden
    check("Smoother: Box wird über Lücke gehalten", len(out2) == 1)
    time.sleep(0.2)
    out3 = smoother.update([])
    check("Smoother: Box verschwindet nach persist_ms", len(out3) == 0)

    # --- 2) Overlay-Zeichnung ---
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    ov = OverlaySettings()
    rendered = draw_overlay(frame, [det], ov)
    # Kreis: Punkt auf dem Umfang (Mitte (320,180), Radius 180) -> (500,180)
    rim = rendered[180, 500]
    check("Overlay: FOV-Kreis in Bildmitte gezeichnet",
          not np.array_equal(rim, np.zeros(3, dtype=np.uint8)),
          f"Pixel Umfang: {rim.tolist()}")
    # Box (10,10)-(50,50): Punkt auf dem linken Rand -> (x=11, y=30)
    check("Overlay: ESP-Box um Erkennung gezeichnet",
          not np.array_equal(rendered[30, 11], np.zeros(3, dtype=np.uint8)))
    check("Overlay: Originalframe unverändert",
          np.array_equal(frame, np.zeros_like(frame)))
    ov.fov_opacity = 0.5
    soft = draw_overlay(frame, [], ov)
    check("Overlay: Transparenz gemischt (Wert ~ halb)",
          approx(float(soft[180, 500][1]), 127.5, tol=1.5),
          f"Pixel: {soft[180, 500].tolist()}")

    # --- 3) End-to-End-Pipeline mit Videoquelle ---
    video_path = make_test_video("tests/_test_video.avi")
    stats = Stats()
    pipeline = Pipeline(FakeDetector(), stats)
    from app.capture import VideoCapture
    pipeline.start(VideoCapture(video_path, loop=True, max_fps=60))
    time.sleep(1.5)
    snap = stats.snapshot()
    frame_now = pipeline.frame_store.get()
    dets = pipeline.detection_store.get()
    check("Pipeline: FrameStore liefert Frames", frame_now is not None)
    check("Pipeline: Erfassungs-FPS > 0", snap["capture_fps"] > 0,
          f"{snap['capture_fps']:.1f} FPS")
    check("Pipeline: Erkennungen liegen vor", len(dets) > 0,
          f"{len(dets)} Box(en), conf={dets[0].conf if dets else 0:.2f}")
    check("Pipeline: keine Fehlermeldung", snap["error"] is None, snap["error"] or "")
    check("Pipeline: läuft-Status korrekt", pipeline.running)
    rendered = draw_overlay(frame_now, dets, OverlaySettings())
    check("Pipeline: gerendertes Frame hat Originalgröße",
          rendered.shape == frame_now.shape)

    # --- 4) Sauberes Stoppen ---
    pipeline.stop()
    time.sleep(0.1)
    check("Pipeline: Worker nach stop() beendet", not pipeline.running)
    check("Pipeline: DetectionStore geleert", len(pipeline.detection_store.get()) == 0)
    check("Pipeline: Status zurückgesetzt", stats.snapshot()["status"] == "Bereit")

    print()
    if failures:
        print(f"{len(failures)} Test(s) fehlgeschlagen: {', '.join(failures)}")
        return 1
    print("Alle Smoke-Tests bestanden.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
