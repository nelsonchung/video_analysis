"""Bounded-memory RAW and FFmpeg readers with explicit timestamp provenance."""
import csv
import json
import subprocess
import tempfile
from decimal import Decimal
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np

from videoqa.schemas import FrameRecord


class CaptureTimes:
    def __init__(self, path):
        # The owning reader closes this resource on completion or cancellation.
        self.handle = Path(path).open(newline="", encoding="utf-8-sig") if path else None
        self.rows = csv.DictReader(self.handle) if self.handle else None
        if self.rows is not None and "timestamp_ns" not in (self.rows.fieldnames or []):
            self.close()
            raise ValueError("Capture timestamp CSV requires timestamp_ns column (integer nanoseconds)")
        self.previous = None

    def next(self, index):
        if self.rows is None:
            return None
        row = next(self.rows, None)
        if row is None:
            raise ValueError(f"Capture timestamp CSV has fewer rows than decoded frames ({index})")
        try:
            value = int(row["timestamp_ns"])
            if "frame_index" in row and int(row["frame_index"]) != index:
                raise ValueError("frame_index must be zero-based and contiguous")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid capture timestamp row {index}: {exc}") from exc
        if value < 0 or (self.previous is not None and value <= self.previous):
            raise ValueError("Capture timestamps must be nonnegative and strictly increasing")
        self.previous = value
        return value

    def finish(self):
        if self.rows is not None and next(self.rows, None) is not None:
            raise ValueError("Capture timestamp CSV has more rows than decoded frames")

    def close(self):
        if self.handle:
            self.handle.close()


class Reader:
    def __init__(self, spec):
        self.spec = spec
        self.metadata = {"backend": "cpu", "timestamp_source": "capture_csv" if spec.capture_timestamps else "none"}
        self.diagnostics = []
        self.times = CaptureTimes(spec.capture_timestamps)
        self.started = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        self.times.close()

    def record(self, index, pixels, pts=None):
        return FrameRecord(self.spec.file_id, index, pixels, width=pixels.shape[1], height=pixels.shape[0],
                           pts_ns=pts, capture_timestamp_ns=self.times.next(index),
                           metadata={"pts_source": "demuxer_pts_not_capture" if pts is not None else "none"})


class RawReader(Reader):
    """8-bit unpacked layouts, with explicit strides and optional plane offsets."""
    def __init__(self, spec):
        super().__init__(spec)
        self.handle = None
        try:
            self._layout()
            size = spec.path.stat().st_size
            if not size or size % self.frame_size:
                raise ValueError(f"RAW size {size} is not a positive multiple of frame_size_bytes={self.frame_size}")
            self.metadata.update(codec="raw", width=spec.width, height=spec.height,
                                 pixel_format=spec.pixel_format, frame_count=size // self.frame_size,
                                 frame_size_bytes=self.frame_size, plane_offsets=self.offsets,
                                 plane_strides=self.strides, header_bytes=spec.header_bytes)
        except Exception:
            self.close()
            raise

    def _layout(self):
        s = self.spec
        if not s.width or not s.height or not s.pixel_format:
            raise ValueError("RAW requires width, height and pixel_format; layout cannot be inferred")
        if not isinstance(s.pixel_format, str):
            raise ValueError("RAW pixel_format must be a string")
        if s.bit_depth != 8 or s.packing != "unpacked":
            raise ValueError("RAW V1 supports only 8-bit unpacked pixels; packed/high-bit-depth layouts require an explicit reader")
        aliases = {"yuyv": "yuyv422", "yuy2": "yuyv422", "gray8": "gray", "yuv420p": "i420"}
        self.fmt = aliases.get(s.pixel_format.lower(), s.pixel_format.lower())
        w, h = s.width, s.height
        if self.fmt in {"nv12", "i420"} and (w % 2 or h % 2):
            raise ValueError("NV12/I420 require even width and height")
        if self.fmt == "yuyv422" and w % 2:
            raise ValueError("YUYV422 requires even width")
        if self.fmt == "nv12":
            self.shapes = [(h, w), (h // 2, w)]
        elif self.fmt == "i420":
            self.shapes = [(h, w), (h // 2, w // 2), (h // 2, w // 2)]
        elif self.fmt in {"bgr24", "rgb24"}:
            self.shapes = [(h, w * 3)]
        elif self.fmt == "yuyv422":
            self.shapes = [(h, w * 2)]
        elif self.fmt in {"gray", "bayer_rggb8", "bayer_bggr8", "bayer_grbg8", "bayer_gbrg8"}:
            self.shapes = [(h, w)]
        else:
            raise ValueError(f"Unsupported RAW pixel_format: {s.pixel_format}")
        def integer(value, name, minimum=0):
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
            return value
        integer(s.header_bytes, "header_bytes")
        if s.stride_bytes is not None:
            integer(s.stride_bytes, "stride_bytes", 1)
        if s.plane_strides is not None:
            if not isinstance(s.plane_strides, list):
                raise ValueError("plane_strides must be a list with one entry per plane")
            if len(s.plane_strides) != len(self.shapes):
                raise ValueError("plane_strides must have one entry per plane")
            self.strides = [integer(x, "plane_strides", 1) for x in s.plane_strides]
        elif s.stride_bytes is not None:
            if self.fmt == "i420":
                if s.stride_bytes % 2:
                    raise ValueError("I420 stride_bytes must be even or supply plane_strides")
                self.strides = [s.stride_bytes, s.stride_bytes // 2, s.stride_bytes // 2]
            else:
                self.strides = [s.stride_bytes] * len(self.shapes)
        else:
            self.strides = [row_bytes for _, row_bytes in self.shapes]
        if any(stride < shape[1] for stride, shape in zip(self.strides, self.shapes)):
            raise ValueError("Plane stride is smaller than active row width")
        if s.plane_offsets is not None:
            if not isinstance(s.plane_offsets, list):
                raise ValueError("plane_offsets must be a list with one entry per plane")
            if len(s.plane_offsets) != len(self.shapes):
                raise ValueError("plane_offsets must have one entry per plane")
            self.offsets = [integer(x, "plane_offsets") for x in s.plane_offsets]
        else:
            self.offsets = []
            end = 0
            for stride, (rows, _) in zip(self.strides, self.shapes):
                self.offsets.append(end)
                end += stride * rows
        intervals = sorted((offset, offset + stride * rows) for offset, stride, (rows, _) in
                           zip(self.offsets, self.strides, self.shapes))
        if any(right[0] < left[1] for left, right in pairwise(intervals)):
            raise ValueError("RAW plane ranges overlap")
        required = s.header_bytes + max(end for _, end in intervals)
        self.frame_size = integer(s.frame_size_bytes, "frame_size_bytes", 1) if s.frame_size_bytes is not None else required
        if self.frame_size < required:
            raise ValueError("frame_size_bytes cannot contain the declared RAW planes/header")

    def __iter__(self):
        if self.started:
            raise RuntimeError("Reader can only be iterated once")
        self.started = True
        s = self.spec
        with s.path.open("rb") as handle:
            self.handle = handle
            index = 0
            while data := handle.read(self.frame_size):
                if len(data) != self.frame_size:
                    raise ValueError("Truncated RAW frame")
                planes = [np.ndarray((rows, row_bytes), np.uint8, data,
                                     offset=s.header_bytes + offset, strides=(stride, 1)).copy()
                          for (rows, row_bytes), offset, stride in zip(self.shapes, self.offsets, self.strides)]
                if self.fmt in {"nv12", "i420"}:
                    packed = np.concatenate([plane.ravel() for plane in planes]).reshape(s.height * 3 // 2, s.width)
                    pixels = cv2.cvtColor(packed, cv2.COLOR_YUV2BGR_NV12 if self.fmt == "nv12" else cv2.COLOR_YUV2BGR_I420)
                elif self.fmt == "yuyv422":
                    pixels = cv2.cvtColor(planes[0].reshape(s.height, s.width, 2), cv2.COLOR_YUV2BGR_YUY2)
                elif self.fmt in {"rgb24", "bgr24"}:
                    pixels = planes[0].reshape(s.height, s.width, 3)
                    if self.fmt == "rgb24":
                        pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
                elif self.fmt == "gray":
                    pixels = cv2.cvtColor(planes[0], cv2.COLOR_GRAY2BGR)
                else:
                    code = {"bayer_rggb8": cv2.COLOR_BayerRGGB2BGR, "bayer_bggr8": cv2.COLOR_BayerBGGR2BGR,
                            "bayer_grbg8": cv2.COLOR_BayerGRBG2BGR, "bayer_gbrg8": cv2.COLOR_BayerGBRG2BGR}[self.fmt]
                    pixels = cv2.cvtColor(planes[0], code)
                yield self.record(index, pixels)
                index += 1
        self.times.finish()

    def close(self):
        if getattr(self, "handle", None):
            self.handle.close()
        super().close()


class FFmpegReader(Reader):
    """Decode BGR pixels and independently stream original per-frame PTS via ffprobe."""
    def __init__(self, spec):
        super().__init__(spec)
        self.process = None
        self.probe = None
        self.error_file = None
        self.probe_errors = None
        try:
            result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams", "-of", "json", str(spec.path)],
                                    capture_output=True, text=True, timeout=60, check=False)
            if result.returncode:
                raise ValueError(f"ffprobe failed: {result.stderr.strip()}")
            if result.stderr.strip():
                self.diagnostics.append(result.stderr.strip())
            streams = json.loads(result.stdout).get("streams", [])
            if not streams:
                raise ValueError("No video stream found")
            stream = streams[0]
            expected = {"h264": "h264", "h265": "hevc", "mjpg": "mjpeg"}[spec.codec]
            if stream.get("codec_name") != expected:
                raise ValueError(f"Codec mismatch: expected {expected}, got {stream.get('codec_name')}")
            self.width, self.height = int(stream["width"]), int(stream["height"])
            if self.width <= 0 or self.height <= 0:
                raise ValueError("Invalid decoded dimensions")
            if ((spec.width is not None and self.width != spec.width) or
                    (spec.height is not None and self.height != spec.height)):
                raise ValueError(f"Resolution mismatch: decoded {self.width}x{self.height}, expected {spec.width}x{spec.height}")
            self.metadata.update(codec=expected, width=self.width, height=self.height,
                                 codec_profile=stream.get("profile"), pixel_format=stream.get("pix_fmt"),
                                 declared_frame_rate=stream.get("avg_frame_rate"), time_base=stream.get("time_base"),
                                 pts_source="demuxer_pts_not_capture")
        except Exception:
            self.close()
            raise

    @staticmethod
    def _read_exact(handle, size):
        chunks = bytearray()
        while len(chunks) < size:
            chunk = handle.read(size - len(chunks))
            if not chunk:
                break
            chunks.extend(chunk)
        return chunks

    def __iter__(self):
        if self.started:
            raise RuntimeError("Reader can only be iterated once")
        self.started = True
        try:
            # close() owns these files because the generator may be cancelled mid-frame.
            self.error_file, self.probe_errors = tempfile.TemporaryFile(), tempfile.TemporaryFile()
            self.process = subprocess.Popen(["ffmpeg", "-v", "error", "-nostdin", "-threads", "1", "-noautorotate", "-i", str(self.spec.path),
                                             "-map", "0:v:0", "-an", "-sn", "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"],
                                            stdout=subprocess.PIPE, stderr=self.error_file)
            self.probe = subprocess.Popen(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
                                           "-show_entries", "frame=pts_time,width,height", "-of", "compact=p=0:nk=0", str(self.spec.path)],
                                          stdout=subprocess.PIPE, stderr=self.probe_errors, text=True)
            index = 0
            while data := self._read_exact(self.process.stdout, self.width * self.height * 3):
                if len(data) != self.width * self.height * 3:
                    raise ValueError("FFmpeg produced a truncated decoded frame")
                # Empty lines and side-data lines are not frame records.
                while True:
                    line = self.probe.stdout.readline()
                    values = dict(part.split("=", 1) for part in line.strip().split("|") if "=" in part)
                    if not line or "width" in values:
                        break
                if not line:
                    raise ValueError("FFprobe frame count does not match decoded frames")
                if int(values["width"]) != self.width or int(values["height"]) != self.height:
                    raise ValueError("Video resolution changed within stream")
                value = values.get("pts_time")
                pts = int(Decimal(value) * 1_000_000_000) if value and value != "N/A" else None
                pixels = np.frombuffer(data, np.uint8).reshape(self.height, self.width, 3)
                yield self.record(index, pixels, pts)
                index += 1
            # Drain probe to avoid blocking on a full pipe if decoder discarded frames.
            extra = False
            for line in self.probe.stdout:
                if "width=" in line:
                    extra = True
            code = self.process.wait(timeout=60)
            probe_code = self.probe.wait(timeout=60)
            self._collect_errors()
            if code or probe_code:
                raise ValueError(f"Decoder failed (ffmpeg={code}, ffprobe={probe_code}): {'; '.join(self.diagnostics)}")
            if extra:
                raise ValueError("FFprobe frame count exceeds decoded frame count")
            if not index:
                raise ValueError("No valid frames decoded")
            self.metadata["frame_count"] = index
            self.times.finish()
        finally:
            self.close()

    def _collect_errors(self):
        for name, handle in (("ffmpeg", self.error_file), ("ffprobe", self.probe_errors)):
            if handle and not handle.closed:
                handle.seek(0)
                value = handle.read(65536).decode("utf-8", errors="replace").strip()
                if value:
                    message = f"{name}: {value}"
                    if message not in self.diagnostics:
                        self.diagnostics.append(message)

    def close(self):
        for process in (getattr(self, "process", None), getattr(self, "probe", None)):
            if process:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                if process.stdout:
                    process.stdout.close()
        self._collect_errors()
        for handle in (getattr(self, "error_file", None), getattr(self, "probe_errors", None)):
            if handle:
                handle.close()
        super().close()
