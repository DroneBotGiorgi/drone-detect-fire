from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import yaml


@dataclass
class ModelConfig:
    path: Path
    conf: float
    iou: float


@dataclass
class CaptureConfig:
    max_fps: int
    bitrate_mbps: int
    use_original_resolution: bool
    resize_width: int
    resize_height: int
    restart_on_eof: bool


@dataclass
class RuntimeConfig:
    window_name: str
    show_metrics: bool
    log_file: Path
    yolo_interval_sec: float
    detection_conf_threshold: float
    detection_iou_threshold: float
    detection_max_det: int
    detection_imgsz: int


@dataclass
class PathsConfig:
    adb_path: Path
    ffmpeg_path: Path


@dataclass
class AppConfig:
    paths: PathsConfig
    capture: CaptureConfig
    runtime: RuntimeConfig
    models_default: str
    models_registry: Dict[str, ModelConfig]


def _resolve(path_value: str, root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def load_config(config_path: Path) -> AppConfig:
    config_path = config_path.resolve()
    root = config_path.parent.parent

    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    paths = PathsConfig(
        adb_path=_resolve(raw["paths"]["adb_path"], root),
        ffmpeg_path=_resolve(raw["paths"]["ffmpeg_path"], root),
    )

    capture = CaptureConfig(
        max_fps=int(raw["capture"]["max_fps"]),
        bitrate_mbps=int(raw["capture"]["bitrate_mbps"]),
        use_original_resolution=bool(raw["capture"]["use_original_resolution"]),
        resize_width=int(raw["capture"]["resize_width"]),
        resize_height=int(raw["capture"]["resize_height"]),
        restart_on_eof=bool(raw["capture"]["restart_on_eof"]),
    )

    runtime = RuntimeConfig(
        window_name=str(raw["runtime"]["window_name"]),
        show_metrics=bool(raw["runtime"]["show_metrics"]),
        log_file=_resolve(raw["runtime"]["log_file"], root),
        yolo_interval_sec=float(raw["runtime"].get("yolo_interval_sec", 0.0)),
        detection_conf_threshold=float(raw["runtime"].get("detection_conf_threshold", 0.7)),
        detection_iou_threshold=float(raw["runtime"].get("detection_iou_threshold", 0.30)),
        detection_max_det=int(raw["runtime"].get("detection_max_det", 12)),
        detection_imgsz=int(raw["runtime"].get("detection_imgsz", 640)),
    )

    registry: Dict[str, ModelConfig] = {}
    for alias, model_raw in raw["models"]["registry"].items():
        registry[str(alias)] = ModelConfig(
            path=_resolve(str(model_raw["path"]), root),
            conf=float(model_raw["conf"]),
            iou=float(model_raw["iou"]),
        )

    return AppConfig(
        paths=paths,
        capture=capture,
        runtime=runtime,
        models_default=str(raw["models"]["default"]),
        models_registry=registry,
    )
