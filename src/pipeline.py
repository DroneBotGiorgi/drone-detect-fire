from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Protocol

import cv2
from ultralytics.engine.results import Results

from config import RuntimeConfig
from detector import YoloDetector


class CaptureSource(Protocol):
    def start(self) -> None: ...
    def read(self): ...
    def stop(self) -> None: ...


@dataclass
class RuntimeState:
    stop_requested: bool = False
    pending_model: Optional[str] = None
    yolo_interval_sec: float = 0.0
    detection_conf_threshold: float = 0.25
    detection_iou_threshold: float = 0.30
    detection_max_det: int = 300
    detection_imgsz: int = 640
    pending_yolo_interval_sec: Optional[float] = None
    pending_detection_conf_threshold: Optional[float] = None
    pending_detection_iou_threshold: Optional[float] = None
    pending_detection_max_det: Optional[int] = None
    pending_detection_imgsz: Optional[int] = None
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

    def request_yolo_interval(self, interval_sec: float) -> None:
        with self.lock:
            self.pending_yolo_interval_sec = max(interval_sec, 0.0)

    def consume_yolo_interval_update(self) -> Optional[float]:
        with self.lock:
            value = self.pending_yolo_interval_sec
            if value is not None:
                self.yolo_interval_sec = value
            self.pending_yolo_interval_sec = None
            return value

    def get_yolo_interval(self) -> float:
        with self.lock:
            return self.yolo_interval_sec

    def request_detection_conf_threshold(self, threshold: float) -> None:
        with self.lock:
            value = min(max(float(threshold), 0.0), 1.0)
            self.pending_detection_conf_threshold = value

    def consume_detection_conf_threshold_update(self) -> Optional[float]:
        with self.lock:
            value = self.pending_detection_conf_threshold
            if value is not None:
                self.detection_conf_threshold = value
            self.pending_detection_conf_threshold = None
            return value

    def get_detection_conf_threshold(self) -> float:
        with self.lock:
            return self.detection_conf_threshold

    def request_detection_iou_threshold(self, threshold: float) -> None:
        with self.lock:
            value = min(max(float(threshold), 0.0), 1.0)
            self.pending_detection_iou_threshold = value

    def consume_detection_iou_threshold_update(self) -> Optional[float]:
        with self.lock:
            value = self.pending_detection_iou_threshold
            if value is not None:
                self.detection_iou_threshold = value
            self.pending_detection_iou_threshold = None
            return value

    def get_detection_iou_threshold(self) -> float:
        with self.lock:
            return self.detection_iou_threshold

    def request_detection_max_det(self, value: int) -> None:
        with self.lock:
            self.pending_detection_max_det = max(int(value), 1)

    def consume_detection_max_det_update(self) -> Optional[int]:
        with self.lock:
            value = self.pending_detection_max_det
            if value is not None:
                self.detection_max_det = value
            self.pending_detection_max_det = None
            return value

    def get_detection_max_det(self) -> int:
        with self.lock:
            return self.detection_max_det

    def request_detection_imgsz(self, value: int) -> None:
        with self.lock:
            self.pending_detection_imgsz = max(int(value), 32)

    def consume_detection_imgsz_update(self) -> Optional[int]:
        with self.lock:
            value = self.pending_detection_imgsz
            if value is not None:
                self.detection_imgsz = value
            self.pending_detection_imgsz = None
            return value

    def get_detection_imgsz(self) -> int:
        with self.lock:
            return self.detection_imgsz

    def should_stop(self) -> bool:
        with self.lock:
            return self.stop_requested


def run_pipeline(
    capture: CaptureSource,
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
    last_result: Optional[Results] = None
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

            interval_update = state.consume_yolo_interval_update()
            if interval_update is not None:
                hz = 0.0 if interval_update == 0.0 else (1.0 / interval_update)
                logger.info("YOLO interval updated: %.3fs (%.2f Hz)", interval_update, hz)

            conf_update = state.consume_detection_conf_threshold_update()
            if conf_update is not None:
                detector.set_conf_threshold(conf_update)
                logger.info("YOLO confidence threshold updated: %.2f", conf_update)

            iou_update = state.consume_detection_iou_threshold_update()
            if iou_update is not None:
                detector.set_iou_threshold(iou_update)
                logger.info("YOLO IOU threshold updated: %.2f", iou_update)

            max_det_update = state.consume_detection_max_det_update()
            if max_det_update is not None:
                detector.set_max_det(max_det_update)
                logger.info("YOLO max_det updated: %d", max_det_update)

            imgsz_update = state.consume_detection_imgsz_update()
            if imgsz_update is not None:
                detector.set_imgsz(imgsz_update)
                logger.info("YOLO imgsz updated: %d", imgsz_update)

            now = time.perf_counter()
            interval = max(state.get_yolo_interval(), 0.0)
            should_infer = interval == 0.0 or (now - last_infer_ts) >= interval

            annotated = frame.copy()

            if should_infer:
                t_start = time.perf_counter()
                last_result, last_detections = detector.infer(frame)
                last_infer_ms = (time.perf_counter() - t_start) * 1000
                last_infer_ts = now
                inf_counter += 1

            detections = last_detections
            if last_result is not None:
                # Use Ultralytics native plotting so drawn boxes match YOLO output exactly.
                annotated = last_result.plot(img=annotated)

            if runtime_cfg.show_metrics:
                now = time.perf_counter()
                cap_fps = cap_counter / max(now - cap_t0, 1e-6)
                inf_fps = inf_counter / max(now - inf_t0, 1e-6)
                wait_ms = max(0.0, interval - (now - last_infer_ts)) * 1000
                model_hz = 0.0 if interval == 0.0 else (1.0 / interval)
                conf_thr = detector.current_conf_threshold()
                iou_thr = detector.current_iou_threshold()
                max_det = detector.current_max_det()
                imgsz = detector.current_imgsz()
                metrics = (
                    f"model={detector.current_alias} "
                    f"det={detections} "
                    f"conf={conf_thr:.2f} "
                    f"iou={iou_thr:.2f} "
                    f"max={max_det} "
                    f"img={imgsz} "
                    f"hz={model_hz:.2f} "
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
