"""Hauptfenster: Videovorschau mit Overlay + Steuerungspanel.

Die Oberfläche läuft im Haupt-Thread (tkinter-Anforderung) und holt sich
per Timer (~60 Hz) das jüngste Frame und die Erkennungen aus der Pipeline.
Alle Regler ändern die Settings-Objekte direkt -> Live-Effekt.
"""

import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from ..capture import ScreenCapture, VideoCapture
from ..config import CaptureSettings, DetectionSettings, OverlaySettings
from ..detection.detector import Detector
from ..overlay import draw_overlay
from ..pipeline import Pipeline

VIEW_W, VIEW_H = 960, 540  # Größe des Vorschau-Canvas


class MainWindow:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("KI-Aimbot-Visualisierer (nur Visualisierung – keine Spiel-Eingriffe)")
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        # gemeinsame, veränderbare Einstellungen
        self.overlay_settings = OverlaySettings()
        self.detection_settings = DetectionSettings()
        self.capture_settings = CaptureSettings()

        self.detector = Detector(self.detection_settings)
        self.stats = None  # wird pro Pipeline-Start erzeugt
        self.pipeline = None
        self._photo = None  # Referenz halten, damit tkinter das Bild nicht freigibt
        self._file_path: str | None = None

        self._build_layout()
        self._refresh()

    # ------------------------------------------------------------ Layout

    def _build_layout(self) -> None:
        main = ttk.Frame(self.root, padding=8)
        main.pack(fill=tk.BOTH, expand=True)

        # Vorschau links
        self.canvas = tk.Canvas(main, width=VIEW_W, height=VIEW_H, bg="#101014",
                                highlightthickness=1, highlightbackground="#333")
        self.canvas.grid(row=0, column=0, sticky="nsew")

        # Steuerung rechts
        panel = ttk.Frame(main, padding=(12, 0, 0, 0))
        panel.grid(row=0, column=1, sticky="ns")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=1)

        # --- Quelle ---
        box_source = ttk.LabelFrame(panel, text="Videoquelle", padding=8)
        box_source.pack(fill=tk.X, pady=(0, 8))
        self.source_var = tk.StringVar(value="screen")
        for value, label in (("screen", "Bildschirm (Echtzeit)"),
                             ("file", "Videodatei"),
                             ("camera", "Webcam")):
            ttk.Radiobutton(box_source, text=label, value=value,
                            variable=self.source_var,
                            command=self._pick_source).pack(anchor=tk.W)
        self.file_label = ttk.Label(box_source, text="Keine Datei gewählt",
                                    foreground="#666")
        self.file_label.pack(anchor=tk.W, pady=(4, 0))
        self.file_button = ttk.Button(box_source, text="Datei auswählen…",
                                      command=self._choose_file)
        self.file_button.pack(anchor=tk.W, pady=(2, 0))

        # --- Start/Stopp ---
        box_run = ttk.Frame(panel)
        box_run.pack(fill=tk.X, pady=(0, 8))
        self.start_button = ttk.Button(box_run, text="▶ Start", command=self._start)
        self.start_button.pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.stop_button = ttk.Button(box_run, text="■ Stopp", command=self._stop,
                                      state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(6, 0))

        # --- FOV-Kreis ---
        box_fov = ttk.LabelFrame(panel, text="FOV-Kreis", padding=8)
        box_fov.pack(fill=tk.X, pady=(0, 8))
        self.fov_enabled = tk.BooleanVar(value=self.overlay_settings.fov_enabled)
        ttk.Checkbutton(box_fov, text="Anzeigen", variable=self.fov_enabled,
                        command=self._apply_overlay).pack(anchor=tk.W)
        self._slider(box_fov, "Größe (Radius)", 20, 400,
                     self.overlay_settings.fov_radius,
                     lambda v: setattr(self.overlay_settings, "fov_radius", int(float(v))))
        self._slider(box_fov, "Linienstärke", 1, 8,
                     self.overlay_settings.fov_thickness,
                     lambda v: setattr(self.overlay_settings, "fov_thickness", int(float(v))))
        self._slider(box_fov, "Transparenz (0–100 %)", 0, 100, 100,
                     lambda v: setattr(self.overlay_settings, "fov_opacity", float(v) / 100.0))

        # --- ESP-Boxen ---
        box_esp = ttk.LabelFrame(panel, text="ESP-Boxen", padding=8)
        box_esp.pack(fill=tk.X, pady=(0, 8))
        self.esp_enabled = tk.BooleanVar(value=self.overlay_settings.esp_enabled)
        ttk.Checkbutton(box_esp, text="Anzeigen", variable=self.esp_enabled,
                        command=self._apply_overlay).pack(anchor=tk.W)
        self._slider(box_esp, "Linienstärke", 1, 8,
                     self.overlay_settings.esp_thickness,
                     lambda v: setattr(self.overlay_settings, "esp_thickness", int(float(v))))
        self._slider(box_esp, "Transparenz (0–100 %)", 0, 100, 100,
                     lambda v: setattr(self.overlay_settings, "esp_opacity", float(v) / 100.0))
        self.show_conf = tk.BooleanVar(value=self.overlay_settings.show_confidence)
        ttk.Checkbutton(box_esp, text="Konfidenz anzeigen", variable=self.show_conf,
                        command=self._apply_overlay).pack(anchor=tk.W)

        # --- Erkennung ---
        box_det = ttk.LabelFrame(panel, text="KI-Erkennung", padding=8)
        box_det.pack(fill=tk.X, pady=(0, 8))
        self._slider(box_det, "Erkennungsschwelle (%)", 5, 95,
                     int(self.detection_settings.conf_threshold * 100),
                     lambda v: setattr(self.detection_settings, "conf_threshold",
                                       float(v) / 100.0))
        self.model_label = ttk.Label(
            box_det, text=f"Modell: {self.detection_settings.model_path}")
        self.model_label.pack(anchor=tk.W, pady=(4, 0))
        ttk.Button(box_det, text="Anderes Modell laden…",
                   command=self._choose_model).pack(anchor=tk.W, pady=(2, 0))

        # --- Leistung ---
        box_stats = ttk.LabelFrame(panel, text="Leistung", padding=8)
        box_stats.pack(fill=tk.X)
        self.stat_labels = {}
        for key, label in (("capture_fps", "Erfassung: – FPS"),
                           ("detect_fps", "Erkennung: – FPS"),
                           ("infer_ms", "Inferenz: – ms"),
                           ("detection_count", "Erkannt: 0")):
            var = tk.StringVar(value=label)
            self.stat_labels[key] = (var, label)
            ttk.Label(box_stats, textvariable=var).pack(anchor=tk.W)

        # --- Status ---
        self.status_var = tk.StringVar(value="Bereit")
        ttk.Label(main, textvariable=self.status_var, foreground="#0a7",
                  anchor=tk.W).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))

    def _slider(self, parent, label: str, from_: int, to: int, initial: int,
                on_change) -> None:
        ttk.Label(parent, text=label).pack(anchor=tk.W, pady=(6, 0))
        scale = ttk.Scale(parent, from_=from_, to=to, orient=tk.HORIZONTAL,
                          command=on_change)
        scale.set(initial)
        scale.pack(fill=tk.X)

    # ------------------------------------------------------------ Aktionen

    def _pick_source(self) -> None:
        if self.source_var.get() == "file" and self._file_path is None:
            self._choose_file()

    def _choose_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Videodatei auswählen",
            filetypes=[("Videodateien", "*.mp4 *.mkv *.avi *.mov *.webm"),
                       ("Alle Dateien", "*.*")])
        if path:
            self._file_path = path
            short = path if len(path) <= 40 else "…" + path[-39:]
            self.file_label.config(text=short, foreground="#000")

    def _choose_model(self) -> None:
        path = filedialog.askopenfilename(
            title="YOLO-Modell (.pt) auswählen",
            filetypes=[("YOLO-Modelle", "*.pt"), ("Alle Dateien", "*.*")])
        if not path:
            return
        self.detection_settings.model_path = path
        self.model_label.config(text=f"Modell: {path}")
        # geladenes Modell verwerfen, damit es neu geladen wird
        self.detector._model = None

    def _apply_overlay(self) -> None:
        s = self.overlay_settings
        s.fov_enabled = self.fov_enabled.get()
        s.esp_enabled = self.esp_enabled.get()
        s.show_confidence = self.show_conf.get()

    def _make_source(self):
        kind = self.source_var.get()
        cs = self.capture_settings
        if kind == "screen":
            return ScreenCapture(monitor_index=cs.monitor_index, region=cs.region,
                                 max_fps=cs.max_fps)
        if kind == "file":
            if not self._file_path:
                raise RuntimeError("Bitte zuerst eine Videodatei auswählen.")
            return VideoCapture(self._file_path, loop=cs.loop_video, max_fps=cs.max_fps)
        if kind == "camera":
            return VideoCapture(0, max_fps=cs.max_fps)
        raise RuntimeError(f"Unbekannte Quelle: {kind}")

    def _start(self) -> None:
        try:
            source = self._make_source()
        except RuntimeError as exc:
            messagebox.showwarning("Hinweis", str(exc))
            return

        self.stats = self.stats or type(self)._new_stats()
        self.stats.reset_runtime()
        self.pipeline = Pipeline(self.detector, self.stats)
        try:
            self.pipeline.start(source)
        except RuntimeError as exc:
            messagebox.showerror("Fehler", str(exc))
            return
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_var.set("Starte … (Modell lädt ggf. beim ersten Mal)")

    @staticmethod
    def _new_stats():
        from ..stats import Stats
        return Stats()

    def _stop(self) -> None:
        if self.pipeline:
            try:
                self.pipeline.stop()
            except Exception as exc:
                messagebox.showerror("Fehler beim Beenden", str(exc))
            self.pipeline = None
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_var.set("Bereit")

    # ------------------------------------------------------------ Anzeige

    def _refresh(self) -> None:
        """Timer-Loop der Oberfläche: Frame holen, Overlay zeichnen, anzeigen."""
        if self.pipeline and self.pipeline.running:
            frame = self.pipeline.frame_store.get()
            detections = self.pipeline.detection_store.get()
            if frame is not None:
                display = draw_overlay(frame, detections, self.overlay_settings)
                self._show_frame(display)
            snap = self.stats.snapshot()
            self._update_stats(snap)
            self.status_var.set(snap["status"])
            if snap["error"]:
                self.status_var.set(f"Fehler: {snap['error']}")
        self.root.after(16, self._refresh)

    def _show_frame(self, frame_bgr) -> None:
        height, width = frame_bgr.shape[:2]
        scale = min(VIEW_W / width, VIEW_H / height)
        new_w, new_h = int(width * scale), int(height * scale)
        resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        # schwarzes Hintergrundbild in Canvas-Größe, Video mittig einbetten
        canvas_img = Image.new("RGB", (VIEW_W, VIEW_H), (16, 16, 20))
        canvas_img.paste(image, ((VIEW_W - new_w) // 2, (VIEW_H - new_h) // 2))
        self._photo = ImageTk.PhotoImage(canvas_img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self._photo, anchor=tk.NW)

    def _update_stats(self, snap: dict) -> None:
        pairs = (
            ("capture_fps", f"Erfassung: {snap['capture_fps']:.0f} FPS"),
            ("detect_fps", f"Erkennung: {snap['detect_fps']:.0f} FPS"),
            ("infer_ms", f"Inferenz: {snap['infer_ms']:.0f} ms"),
            ("detection_count", f"Erkannt: {snap['detection_count']}"),
        )
        for key, text in pairs:
            var, _ = self.stat_labels[key]
            var.set(text)

    # ------------------------------------------------------------ Beenden

    def _on_close(self) -> None:
        try:
            if self.pipeline:
                self.pipeline.shutdown()
        finally:
            self.root.destroy()
