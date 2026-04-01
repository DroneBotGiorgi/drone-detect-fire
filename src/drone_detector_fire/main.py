from __future__ import annotations

import argparse
import logging
import threading
from pathlib import Path

from .capture import ADBFFmpegCapture, query_device_resolution
from .config import load_config
from .detector import YoloDetector
from .gui import ControlPanel
from .pipeline import RuntimeState, run_pipeline


def _setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def cli_main() -> int:
    parser = argparse.ArgumentParser(description="Low latency Android stream YOLO detector")
    parser.add_argument("--config", type=Path, default=Path("config/settings.yaml"))
    parser.add_argument("--gui", action="store_true", help="Enable Tkinter control panel")
    args = parser.parse_args()

    cfg = load_config(args.config)
    _setup_logging(cfg.runtime.log_file)

    if cfg.capture.use_original_resolution:
        width, height = query_device_resolution(cfg.paths.adb_path)
    else:
        width, height = cfg.capture.resize_width, cfg.capture.resize_height

    detector = YoloDetector(cfg.models_registry, cfg.models_default)
    capture = ADBFFmpegCapture(
        adb_path=cfg.paths.adb_path,
        ffmpeg_path=cfg.paths.ffmpeg_path,
        width=width,
        height=height,
        bitrate_mbps=cfg.capture.bitrate_mbps,
        max_fps=cfg.capture.max_fps,
        restart_on_eof=cfg.capture.restart_on_eof,
    )
    state = RuntimeState()

    if not args.gui:
        run_pipeline(capture, detector, cfg.runtime, state)
        return 0

    worker = threading.Thread(
        target=run_pipeline,
        args=(capture, detector, cfg.runtime, state),
        daemon=True,
    )
    worker.start()

    panel = ControlPanel(state, detector.aliases(), detector.current_alias)
    panel.run()

    worker.join(timeout=5)
    return 0
