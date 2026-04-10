from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
from ultralytics.engine.results import Results
from ultralytics import YOLO

from config import ModelConfig


class YoloDetector:
    def __init__(self, model_registry: Dict[str, ModelConfig], default_alias: str) -> None:
        if default_alias not in model_registry:
            raise ValueError(f"Default model alias not found: {default_alias}")

        self._registry = model_registry
        self._loaded: Dict[str, YOLO] = {}
        self._current_alias = default_alias
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._conf_override: Optional[float] = None
        self._iou_override: Optional[float] = None
        self._max_det_override: Optional[int] = None
        self._imgsz_override: Optional[int] = None

        self._ensure_loaded(default_alias)

    @property
    def current_alias(self) -> str:
        return self._current_alias

    def aliases(self) -> list[str]:
        return list(self._registry.keys())

    def set_conf_threshold(self, value: float) -> None:
        self._conf_override = min(max(float(value), 0.0), 1.0)

    def current_conf_threshold(self) -> float:
        if self._conf_override is not None:
            return self._conf_override
        return self._registry[self._current_alias].conf

    def set_iou_threshold(self, value: float) -> None:
        self._iou_override = min(max(float(value), 0.0), 1.0)

    def current_iou_threshold(self) -> float:
        if self._iou_override is not None:
            return self._iou_override
        return self._registry[self._current_alias].iou

    def set_max_det(self, value: int) -> None:
        self._max_det_override = max(int(value), 1)

    def current_max_det(self) -> int:
        if self._max_det_override is not None:
            return self._max_det_override
        return 300

    def set_imgsz(self, value: int) -> None:
        self._imgsz_override = max(int(value), 32)

    def current_imgsz(self) -> int:
        if self._imgsz_override is not None:
            return self._imgsz_override
        return 640

    def switch_model(self, alias: str) -> None:
        if alias not in self._registry:
            raise ValueError(f"Model alias not found: {alias}")
        self._ensure_loaded(alias)
        self._current_alias = alias

    def _ensure_loaded(self, alias: str) -> None:
        if alias in self._loaded:
            return

        model_path = self._registry[alias].path
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model = YOLO(str(model_path))
        self._loaded[alias] = model

    def infer(self, frame) -> tuple[Results, int]:
        alias = self._current_alias
        model_cfg = self._registry[alias]
        model = self._loaded[alias]

        conf = self._conf_override if self._conf_override is not None else model_cfg.conf
        iou = self._iou_override if self._iou_override is not None else model_cfg.iou
        max_det = self._max_det_override if self._max_det_override is not None else 300
        imgsz = self._imgsz_override if self._imgsz_override is not None else 640

        result = model.predict(
            source=frame,
            conf=conf,
            iou=iou,
            max_det=max_det,
            imgsz=imgsz,
            device=self._device,
            verbose=False,
        )[0]
        detections = 0 if result.boxes is None else len(result.boxes)
        return result, detections
