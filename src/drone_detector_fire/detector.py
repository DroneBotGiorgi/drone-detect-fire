from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from ultralytics import YOLO

from .config import ModelConfig


class YoloDetector:
    def __init__(self, model_registry: Dict[str, ModelConfig], default_alias: str) -> None:
        if default_alias not in model_registry:
            raise ValueError(f"Default model alias not found: {default_alias}")

        self._registry = model_registry
        self._loaded: Dict[str, YOLO] = {}
        self._current_alias = default_alias
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self._ensure_loaded(default_alias)

    @property
    def current_alias(self) -> str:
        return self._current_alias

    def aliases(self) -> list[str]:
        return list(self._registry.keys())

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

    def infer_and_annotate(self, frame):
        alias = self._current_alias
        model_cfg = self._registry[alias]
        model = self._loaded[alias]

        result = model.predict(
            source=frame,
            conf=model_cfg.conf,
            iou=model_cfg.iou,
            device=self._device,
            verbose=False,
        )[0]

        annotated = result.plot()
        detections = 0 if result.boxes is None else len(result.boxes)
        return annotated, detections
