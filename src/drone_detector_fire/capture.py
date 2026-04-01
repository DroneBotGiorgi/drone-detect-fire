from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


def query_device_resolution(adb_path: Path) -> Tuple[int, int]:
    command = [str(adb_path), "shell", "wm", "size"]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError("Unable to query device resolution via adb.")

    match = re.search(r"(\d+)x(\d+)", result.stdout)
    if not match:
        raise RuntimeError("Unable to parse device resolution.")

    return int(match.group(1)), int(match.group(2))


class ADBFFmpegCapture:
    def __init__(
        self,
        adb_path: Path,
        ffmpeg_path: Path,
        width: int,
        height: int,
        bitrate_mbps: int,
        max_fps: int,
        restart_on_eof: bool,
    ) -> None:
        self.adb_path = adb_path
        self.ffmpeg_path = ffmpeg_path
        self.width = width
        self.height = height
        self.bitrate_mbps = bitrate_mbps
        self.max_fps = max_fps
        self.restart_on_eof = restart_on_eof

        self._adb_proc: Optional[subprocess.Popen] = None
        self._ffmpeg_proc: Optional[subprocess.Popen] = None
        self._frame_size = self.width * self.height * 3
        self._mode = "screenrecord"

    @property
    def mode(self) -> str:
        return self._mode

    def _screenrecord_available(self) -> bool:
        probe = subprocess.run(
            [str(self.adb_path), "exec-out", "screenrecord", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        text = (probe.stdout or "") + "\n" + (probe.stderr or "")
        lowered = text.lower()
        if "inaccessible or not found" in lowered:
            return False
        if "not found" in lowered and "screenrecord" in lowered:
            return False
        return True

    def _build_adb_cmd(self) -> list[str]:
        return [
            str(self.adb_path),
            "exec-out",
            "screenrecord",
            "--output-format=h264",
            "--bit-rate",
            str(self.bitrate_mbps * 1_000_000),
            "--size",
            f"{self.width}x{self.height}",
            "-",
        ]

    def _build_ffmpeg_cmd(self) -> list[str]:
        return [
            str(self.ffmpeg_path),
            "-hide_banner",
            "-loglevel",
            "error",
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
            "-probesize",
            "32",
            "-analyzeduration",
            "0",
            "-f",
            "h264",
            "-i",
            "pipe:0",
            "-pix_fmt",
            "bgr24",
            "-f",
            "rawvideo",
            "-r",
            str(self.max_fps),
            "pipe:1",
        ]

    def start(self) -> None:
        if not self._screenrecord_available():
            self._mode = "screencap"
            return

        self._mode = "screenrecord"
        adb_cmd = self._build_adb_cmd()
        ffmpeg_cmd = self._build_ffmpeg_cmd()

        self._adb_proc = subprocess.Popen(
            adb_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        self._ffmpeg_proc = subprocess.Popen(
            ffmpeg_cmd,
            stdin=self._adb_proc.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

    def read(self) -> Optional[np.ndarray]:
        if self._mode == "screencap":
            result = subprocess.run(
                [str(self.adb_path), "exec-out", "screencap", "-p"],
                capture_output=True,
                check=False,
            )
            if result.returncode != 0 or not result.stdout:
                return None

            png = np.frombuffer(result.stdout, dtype=np.uint8)
            frame = cv2.imdecode(png, cv2.IMREAD_COLOR)
            return frame

        if not self._ffmpeg_proc or not self._ffmpeg_proc.stdout:
            return None

        data = self._ffmpeg_proc.stdout.read(self._frame_size)
        if len(data) != self._frame_size:
            if self.restart_on_eof:
                self.stop()
                self.start()
                return None
            return None

        frame = np.frombuffer(data, dtype=np.uint8)
        return frame.reshape((self.height, self.width, 3))

    def stop(self) -> None:
        for proc in (self._ffmpeg_proc, self._adb_proc):
            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    proc.kill()

        self._ffmpeg_proc = None
        self._adb_proc = None
