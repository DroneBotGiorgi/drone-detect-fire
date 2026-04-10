from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from pipeline import RuntimeState


class ControlPanel:
    def __init__(
        self,
        state: RuntimeState,
        aliases: list[str],
        current_alias: str,
        current_conf_threshold: float,
        current_iou_threshold: float,
        current_max_det: int,
        current_imgsz: int,
    ) -> None:
        self.state = state
        self._hz_after_id: str | None = None
        self._conf_after_id: str | None = None
        self._iou_after_id: str | None = None
        self._max_det_after_id: str | None = None
        self._imgsz_after_id: str | None = None

        self.root = tk.Tk()
        self.root.title("Detector Control")
        self.root.geometry("420x560")
        self.root.resizable(False, False)

        self.model_var = tk.StringVar(value=current_alias)
        initial_interval = max(self.state.get_yolo_interval(), 0.0)
        initial_hz = 0.0 if initial_interval == 0.0 else (1.0 / initial_interval)
        self.hz_var = tk.StringVar(value=f"{initial_hz:.2f}")
        self.conf_var = tk.StringVar(value=f"{current_conf_threshold:.2f}")
        self.iou_var = tk.StringVar(value=f"{current_iou_threshold:.2f}")
        self.max_det_var = tk.StringVar(value=str(current_max_det))
        self.imgsz_var = tk.StringVar(value=str(current_imgsz))
        self.status_var = tk.StringVar(value="Auto-apply attivo")

        frame = ttk.Frame(self.root, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="Modello YOLO attivo").pack(anchor=tk.W)
        self.combo = ttk.Combobox(frame, textvariable=self.model_var, values=aliases, state="readonly")
        self.combo.pack(fill=tk.X, pady=(6, 12))
        self.combo.bind("<<ComboboxSelected>>", self._on_model_selected)

        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(10, 10))
        ttk.Label(frame, text="Frequenza YOLO (Hz, 0 = ogni frame)").pack(anchor=tk.W)
        self.hz_entry = ttk.Entry(frame, textvariable=self.hz_var)
        self.hz_entry.pack(fill=tk.X, pady=(6, 8))
        self.hz_entry.bind("<KeyRelease>", self._on_hz_typed)
        self.hz_entry.bind("<FocusOut>", self._apply_frequency_now)

        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(10, 10))
        ttk.Label(frame, text="Soglia riconoscimento YOLO (conf, 0..1)").pack(anchor=tk.W)
        self.conf_entry = ttk.Entry(frame, textvariable=self.conf_var)
        self.conf_entry.pack(fill=tk.X, pady=(6, 8))
        self.conf_entry.bind("<KeyRelease>", self._on_conf_typed)
        self.conf_entry.bind("<FocusOut>", self._apply_confidence_now)

        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(10, 10))
        ttk.Label(frame, text="Soglia NMS IOU (0..1, piu bassa = meno overlap)").pack(anchor=tk.W)
        self.iou_entry = ttk.Entry(frame, textvariable=self.iou_var)
        self.iou_entry.pack(fill=tk.X, pady=(6, 8))
        self.iou_entry.bind("<KeyRelease>", self._on_iou_typed)
        self.iou_entry.bind("<FocusOut>", self._apply_iou_now)

        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(10, 10))
        ttk.Label(frame, text="Max detection per frame (max_det)").pack(anchor=tk.W)
        self.max_det_entry = ttk.Entry(frame, textvariable=self.max_det_var)
        self.max_det_entry.pack(fill=tk.X, pady=(6, 8))
        self.max_det_entry.bind("<KeyRelease>", self._on_max_det_typed)
        self.max_det_entry.bind("<FocusOut>", self._apply_max_det_now)

        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(10, 10))
        ttk.Label(frame, text="Risoluzione inferenza YOLO (imgsz, consigliato multiplo di 32)").pack(anchor=tk.W)
        self.imgsz_entry = ttk.Entry(frame, textvariable=self.imgsz_var)
        self.imgsz_entry.pack(fill=tk.X, pady=(6, 8))
        self.imgsz_entry.bind("<KeyRelease>", self._on_imgsz_typed)
        self.imgsz_entry.bind("<FocusOut>", self._apply_imgsz_now)

        ttk.Label(frame, textvariable=self.status_var).pack(anchor=tk.W, pady=(4, 8))

        ttk.Button(frame, text="Stop Pipeline", command=self._stop).pack(fill=tk.X, pady=(8, 0))

        ttk.Label(frame, text="Hotkeys video: 1..9 switch | q quit").pack(anchor=tk.W, pady=(10, 0))

        self.root.protocol("WM_DELETE_WINDOW", self._stop)

    def _switch(self) -> None:
        self.state.request_model_switch(self.model_var.get())
        self.status_var.set(f"Modello impostato: {self.model_var.get()}")

    def _on_model_selected(self, _event) -> None:
        self._switch()

    def _apply_frequency(self) -> bool:
        raw = self.hz_var.get().strip().replace(",", ".")
        try:
            hz = float(raw)
        except ValueError:
            self.status_var.set("Valore Hz non valido")
            return False

        if hz < 0:
            self.status_var.set("La frequenza non puo essere negativa")
            return False

        interval = 0.0 if hz == 0.0 else (1.0 / hz)
        self.state.request_yolo_interval(interval)
        self.status_var.set(f"Frequenza applicata: {hz:.2f} Hz")
        return True

    def _on_hz_typed(self, _event) -> None:
        if self._hz_after_id is not None:
            self.root.after_cancel(self._hz_after_id)
        self._hz_after_id = self.root.after(400, self._apply_frequency)

    def _apply_frequency_now(self, _event) -> None:
        if self._hz_after_id is not None:
            self.root.after_cancel(self._hz_after_id)
            self._hz_after_id = None
        self._apply_frequency()

    def _apply_confidence(self) -> bool:
        raw = self.conf_var.get().strip().replace(",", ".")
        try:
            conf = float(raw)
        except ValueError:
            self.status_var.set("Valore soglia non valido")
            return False

        if conf < 0.0 or conf > 1.0:
            self.status_var.set("La soglia deve essere tra 0 e 1")
            return False

        self.state.request_detection_conf_threshold(conf)
        self.status_var.set(f"Soglia applicata: {conf:.2f}")
        return True

    def _on_conf_typed(self, _event) -> None:
        if self._conf_after_id is not None:
            self.root.after_cancel(self._conf_after_id)
        self._conf_after_id = self.root.after(400, self._apply_confidence)

    def _apply_confidence_now(self, _event) -> None:
        if self._conf_after_id is not None:
            self.root.after_cancel(self._conf_after_id)
            self._conf_after_id = None
        self._apply_confidence()

    def _apply_iou(self) -> bool:
        raw = self.iou_var.get().strip().replace(",", ".")
        try:
            iou = float(raw)
        except ValueError:
            self.status_var.set("Valore IOU non valido")
            return False

        if iou < 0.0 or iou > 1.0:
            self.status_var.set("IOU deve essere tra 0 e 1")
            return False

        self.state.request_detection_iou_threshold(iou)
        self.status_var.set(f"IOU applicato: {iou:.2f}")
        return True

    def _on_iou_typed(self, _event) -> None:
        if self._iou_after_id is not None:
            self.root.after_cancel(self._iou_after_id)
        self._iou_after_id = self.root.after(400, self._apply_iou)

    def _apply_iou_now(self, _event) -> None:
        if self._iou_after_id is not None:
            self.root.after_cancel(self._iou_after_id)
            self._iou_after_id = None
        self._apply_iou()

    def _apply_max_det(self) -> bool:
        raw = self.max_det_var.get().strip()
        try:
            value = int(raw)
        except ValueError:
            self.status_var.set("Valore max_det non valido")
            return False

        if value < 1 or value > 300:
            self.status_var.set("max_det deve essere tra 1 e 300")
            return False

        self.state.request_detection_max_det(value)
        self.status_var.set(f"max_det applicato: {value}")
        return True

    def _on_max_det_typed(self, _event) -> None:
        if self._max_det_after_id is not None:
            self.root.after_cancel(self._max_det_after_id)
        self._max_det_after_id = self.root.after(400, self._apply_max_det)

    def _apply_max_det_now(self, _event) -> None:
        if self._max_det_after_id is not None:
            self.root.after_cancel(self._max_det_after_id)
            self._max_det_after_id = None
        self._apply_max_det()

    def _apply_imgsz(self) -> bool:
        raw = self.imgsz_var.get().strip()
        try:
            value = int(raw)
        except ValueError:
            self.status_var.set("Valore imgsz non valido")
            return False

        if value < 32 or value > 2048:
            self.status_var.set("imgsz deve essere tra 32 e 2048")
            return False

        if value % 32 != 0:
            self.status_var.set("imgsz consigliato multiplo di 32")

        self.state.request_detection_imgsz(value)
        self.status_var.set(f"imgsz applicato: {value}")
        return True

    def _on_imgsz_typed(self, _event) -> None:
        if self._imgsz_after_id is not None:
            self.root.after_cancel(self._imgsz_after_id)
        self._imgsz_after_id = self.root.after(400, self._apply_imgsz)

    def _apply_imgsz_now(self, _event) -> None:
        if self._imgsz_after_id is not None:
            self.root.after_cancel(self._imgsz_after_id)
            self._imgsz_after_id = None
        self._apply_imgsz()

    def _stop(self) -> None:
        self.state.request_stop()
        self.root.after(100, self.root.destroy)

    def run(self) -> None:
        self.root.mainloop()
