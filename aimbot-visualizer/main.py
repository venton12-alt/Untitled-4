"""Einstiegspunkt der KI-Aimbot-Visualisierung.

Startet nur die Visualisierung: Bildschirmaufnahme, KI-Erkennung und
Overlay-Zeichnung. Es gibt keinerlei Maus-, Tastatur- oder Spiel-Steuerung.
"""

import sys
import tkinter as tk

from app.ui.main_window import MainWindow


def main() -> int:
    root = tk.Tk()
    try:
        MainWindow(root)
    except Exception as exc:
        root.destroy()
        print(f"Fehler beim Start: {exc}", file=sys.stderr)
        return 1
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
