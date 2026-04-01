from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2

from .capture import ADBFFmpegCapture
from .config import RuntimeConfig
from .detector import YoloDetector


@dataclass
class RuntimeState:
    stop_requested: bool = False
    pending_model: Optional[str] = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def request_stop(self) -> None:
        with self.lock:
            self.stop_requested = True

    def request_model_switch(self, alias: str) -> None:
        with self.lock:
            self.pending_model = alias

    def consume_model_switch(self) -> Optional[str]:
        with self.lock:
            alias = self.pending_model
            self.pending_model = None
            return alias

    def should_stop(self) -> bool:
        with self.lock:
            return self.stop_requested


def run_pipeline(
    capture: ADBFFmpegCapture,
    detector: YoloDetector,
    runtime_cfg: RuntimeConfig,
    state: RuntimeState,
) -> None:
    logger = logging.getLogger("pipeline")
    capture.start()

    win_name = runtime_cfg.window_name
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    aliases = detector.aliases()
    key_map = {str(i + 1): alias for i, alias in enumerate(aliases[:9])}

    cap_counter = 0
    inf_counter = 0
    cap_t0 = time.perf_counter()
    inf_t0 = time.perf_counter()
    last_infer_ts = 0.0
    last_detections = 0
    last_infer_ms = 0.0

    try:
        while not state.should_stop():
            frame = capture.read()
            if frame is None:
                continue

            cap_counter += 1
            requested_alias = state.consume_model_switch()
            if requested_alias:
                detector.switch_model(requested_alias)
                logger.info("Model switched from GUI: %s", requested_alias)

            now = time.perf_counter()
            interval = max(runtime_cfg.yolo_interval_sec, 0.0)
            should_infer = interval == 0.0 or (now - last_infer_ts) >= interval

            if should_infer:
                t_start = time.perf_counter()
                annotated, detections = detector.infer_and_annotate(frame)
                last_infer_ms = (time.perf_counter() - t_start) * 1000
                last_detections = detections
                last_infer_ts = now
                inf_counter += 1
            else:
                annotated = frame
                detections = last_detections

            if runtime_cfg.show_metrics:
                now = time.perf_counter()
                cap_fps = cap_counter / max(now - cap_t0, 1e-6)
                inf_fps = inf_counter / max(now - inf_t0, 1e-6)
                wait_ms = max(0.0, interval - (now - last_infer_ts)) * 1000
                metrics = (
                    f"model={detector.current_alias} "
                    f"det={detections} "
                    f"cap_fps={cap_fps:.1f} "
                    f"inf_fps={inf_fps:.1f} "
                    f"lat={last_infer_ms:.1f}ms "
                    f"next={wait_ms:.0f}ms"
                )
                cv2.putText(
                    annotated,
                    metrics,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow(win_name, annotated)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                state.request_stop()
                break

            if ord("1") <= key <= ord("9"):
                pressed = chr(key)
                if pressed in key_map:
                    new_alias = key_map[pressed]
                    detector.switch_model(new_alias)
                    logger.info("Model switched by hotkey %s -> %s", pressed, new_alias)

    finally:
        capture.stop()
        cv2.destroyAllWindows()
