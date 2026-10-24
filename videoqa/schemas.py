"""Shared CPU baseline contracts."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

VERSION = "1.0.0"
STATUSES = ("PASS", "FAIL", "SUSPECT", "UNVERIFIABLE", "ERROR")


@dataclass
class FileSpec:
    path: Path
    codec: str
    width: int | None = None
    height: int | None = None
    file_id: str = ""
    expected_fps: float = 30.0
    capture_timestamps: Path | None = None
    pixel_format: str | None = None
    bit_depth: int = 8
    stride_bytes: int | None = None
    plane_offsets: list[int] | None = None
    plane_strides: list[int] | None = None
    frame_size_bytes: int | None = None
    header_bytes: int = 0
    profile: str | None = None
    scenario: str = "unspecified"
    packing: str = "unpacked"

    def __post_init__(self):
        self.path = Path(self.path)
        self.file_id = self.file_id or self.path.stem
        if not isinstance(self.file_id, str):
            raise ValueError("file_id must be a string")
        if self.pixel_format is not None and not isinstance(self.pixel_format, str):
            raise ValueError("pixel_format must be a string")
        for name in ("plane_offsets", "plane_strides"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, list):
                raise ValueError(f"{name} must be a list")
        if self.capture_timestamps is not None:
            self.capture_timestamps = Path(self.capture_timestamps)
        if self.codec not in {"raw", "h264", "h265", "mjpg"}:
            raise ValueError(f"Unsupported codec: {self.codec}")
        if not np.isfinite(self.expected_fps) or self.expected_fps <= 0:
            raise ValueError("expected_fps must be positive and finite")
        for name in ("width", "height"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
                raise ValueError(f"{name} must be a positive integer")


@dataclass
class FrameRecord:
    file_id: str
    frame_index: int
    pixels: np.ndarray
    pixel_format: str = "bgr24"
    width: int = 0
    height: int = 0
    pts_ns: int | None = None
    capture_timestamp_ns: int | None = None
    decode_status: str = "ok"
    metadata: dict[str, Any] = field(default_factory=dict)
