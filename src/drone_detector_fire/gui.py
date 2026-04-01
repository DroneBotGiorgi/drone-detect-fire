from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from .pipeline import RuntimeState


class ControlPanel:
    def __init__(self, state: RuntimeState, aliases: list[str], current_alias: str) -> None:
        self.state = state

        self.root = tk.Tk()
        self.root.title("Detector Control")
        self.root.geometry("360x180")
        self.root.resizable(False, False)

        self.model_var = tk.StringVar(value=current_alias)

        frame = ttk.Frame(self.root, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="Modello YOLO attivo").pack(anchor=tk.W)
        self.combo = ttk.Combobox(frame, textvariable=self.model_var, values=aliases, state="readonly")
        self.combo.pack(fill=tk.X, pady=(6, 12))

        ttk.Button(frame, text="Switch Modello", command=self._switch).pack(fill=tk.X)
        ttk.Button(frame, text="Stop Pipeline", command=self._stop).pack(fill=tk.X, pady=(8, 0))

        ttk.Label(frame, text="Hotkeys video: 1..9 switch | q quit").pack(anchor=tk.W, pady=(10, 0))

        self.root.protocol("WM_DELETE_WINDOW", self._stop)

    def _switch(self) -> None:
        self.state.request_model_switch(self.model_var.get())

    def _stop(self) -> None:
        self.state.request_stop()
        self.root.after(100, self.root.destroy)

    def run(self) -> None:
        self.root.mainloop()
