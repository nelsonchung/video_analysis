"""Streaming CPU metrics with explicit evidence and confidence limits."""
import math
from collections import deque
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import yaml

from .schemas import VERSION, FileSpec, FrameRecord

DEFAULT_PROFILE = {
    "mode": "generic", "calibrated": False, "roi": None, "max_events": 20,
    "fps": {"tolerance_ratio": 0.05, "gap_factor": 1.5, "window_seconds": 1.0},
    "freeze": {"difference_threshold": 0.5, "min_seconds": 1.0},
    "green": {"min_ratio": 0.2, "hue_min": 35, "hue_max": 85,
              "min_saturation": 150, "min_value": 40},
    "ghost": {"difference_threshold": 5.0, "return_ratio": 0.25},
    "blur": {"min_laplacian_variance": 40.0},
    "color": {"reference_rgb": None, "max_rgb_distance": 30.0},
}


def load_profile(path=None):
    """Load a YAML profile, rejecting misspelled or invalid thresholds."""
    if path is None:
        return deepcopy(DEFAULT_PROFILE)
    with Path(path).open(encoding="utf-8") as handle:
        supplied = yaml.safe_load(handle)
    return validate_profile(supplied)


def validate_profile(supplied):
    if not isinstance(supplied, dict):
        raise ValueError("profile must be a mapping")
    result = deepcopy(DEFAULT_PROFILE)
    for key, value in supplied.items():
        if key not in result:
            raise ValueError(f"unknown profile setting: {key}")
        if isinstance(result[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"profile {key} must be a mapping")
            for field, threshold in value.items():
                if field not in result[key]:
                    raise ValueError(f"unknown profile setting: {key}.{field}")
                result[key][field] = threshold
        else:
            result[key] = value
    if result["mode"] not in ("generic", "reference"):
        raise ValueError("mode must be generic or reference")
    if not isinstance(result["calibrated"], bool):
        raise ValueError("calibrated must be a boolean")
    maximum = result["max_events"]
    if isinstance(maximum, bool) or not isinstance(maximum, int) or not 1 <= maximum <= 1000:
        raise ValueError("max_events must be an integer in 1..1000")
    roi = result["roi"]
    if roi is not None and (not isinstance(roi, list) or len(roi) != 4
                            or any(isinstance(v, bool) or not isinstance(v, int) for v in roi)
                            or min(roi[:2]) < 0 or min(roi[2:]) <= 0):
        raise ValueError("roi must be [x, y, width, height] with nonnegative origin and positive dimensions")
    for group in ("fps", "freeze", "green", "ghost", "blur", "color"):
        for name, value in result[group].items():
            if name == "reference_rgb":
                if value is not None and (not isinstance(value, list) or len(value) != 3
                    or any(isinstance(v, bool) or not isinstance(v, (int, float))
                           or not math.isfinite(v) or not 0 <= v <= 255 for v in value)):
                    raise ValueError("reference_rgb must be three finite RGB values in 0..255")
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{group}.{name} must be a finite nonnegative number")
    for group, name in (("fps", "tolerance_ratio"), ("green", "min_ratio"), ("ghost", "return_ratio")):
        if not 0 <= result[group][name] <= 1:
            raise ValueError(f"{group}.{name} must be in 0..1")
    if result["fps"]["gap_factor"] <= 1 or result["fps"]["window_seconds"] <= 0 or result["freeze"]["min_seconds"] <= 0:
        raise ValueError("gap_factor must exceed 1; window_seconds and min_seconds must be positive")
    green = result["green"]
    if not 0 <= green["hue_min"] <= green["hue_max"] <= 179:
        raise ValueError("green hue limits must be ordered within 0..179")
    if any(green[key] > 255 for key in ("min_saturation", "min_value")):
        raise ValueError("green saturation/value thresholds must be within 0..255")
    return result


class Analyzer:
    """Keep two previous images and bounded events, rather than a full video."""

    def __init__(self, spec: FileSpec, profile: dict, output_dir: Path):
        self.spec = spec
        self.profile = validate_profile(profile)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.count = 0
        self.events = []
        self.event_counts = dict.fromkeys(("freeze", "green", "ghost", "blur", "color", "fps"), 0)
        self.active = {}
        self.previous = None
        self.previous_gray = None
        self.older_gray = None
        self.previous_difference = None
        self.pending = []
        self.freeze_start = None
        self.capture_first = self.capture_last = None
        self.pts_first = self.pts_last = None
        self.capture_count = self.pts_count = 0
        self.capture_invalid = self.pts_invalid = False
        self.gaps = self.drops = 0
        self.local_times = deque()
        self.min_local_fps = None
        self.min_sharpness = None
        self.max_green_ratio = 0.0
        self.rgb_sum = np.zeros(3, dtype=float)
        self.hsv_sum = np.zeros(3, dtype=float)
        self.ycrcb_sum = np.zeros(3, dtype=float)
        self.max_color_distance = 0.0
        self.decode_bad = 0
        self.last_frame_index = None

    @property
    def candidate_status(self):
        return "FAIL" if self.profile["mode"] == "reference" and self.profile["calibrated"] else "SUSPECT"

    def _time(self, frame):
        return frame.capture_timestamp_ns if frame.capture_timestamp_ns is not None else frame.pts_ns

    def _save(self, frame, role):
        directory = self.output_dir / "evidence"
        directory.mkdir(exist_ok=True)
        name = f"frame_{frame.frame_index:08d}_{role}.jpg"
        target = directory / name
        if not cv2.imwrite(str(target), frame.pixels):
            raise OSError(f"could not write evidence: {target}")
        return f"evidence/{name}"

    def _event(self, detector, frame, metrics, start=None):
        self.event_counts[detector] += 1
        if sum(event["detector"] == detector for event in self.events) >= self.profile["max_events"]:
            return None
        event = {
            "detector": detector, "status": "FAIL" if detector == "fps" else self.candidate_status,
            "start_frame": frame.frame_index if start is None else start[0],
            "end_frame": frame.frame_index,
            "start_time_ns": self._time(frame) if start is None else start[1],
            "end_time_ns": self._time(frame), "metrics": metrics,
            "thresholds": deepcopy(self.profile[detector]), "version": VERSION,
            "confidence": "high" if detector == "fps" else "low" if self.candidate_status == "SUSPECT" else "medium",
            "evidence": [],
        }
        if self.previous is not None:
            event["evidence"].append(self._save(self.previous, "before"))
        event["evidence"].append(self._save(frame, "event"))
        self.pending.append(event)
        self.events.append(event)
        return event

    def _candidate(self, detector, condition, frame, metrics):
        if condition:
            if detector not in self.active:
                self.active[detector] = self._event(detector, frame, metrics)
            event = self.active[detector]
            if event is not None:
                event["end_frame"] = frame.frame_index
                event["end_time_ns"] = self._time(frame)
        else:
            self.active.pop(detector, None)

    def _timestamps(self, frame):
        capture = frame.capture_timestamp_ns
        pts = frame.pts_ns
        if capture is not None:
            self.capture_count += 1
            if self.capture_first is None:
                self.capture_first = capture
            if self.capture_last is not None:
                gap = capture - self.capture_last
                if gap <= 0:
                    self.capture_invalid = True
                elif gap > self.profile["fps"]["gap_factor"] * 1e9 / self.spec.expected_fps:
                    self.gaps += 1
                    missing = max(1, round(gap * self.spec.expected_fps / 1e9) - 1)
                    self.drops += missing
                    self._event("fps", frame, {"gap_ns": gap, "estimated_dropped_frames": missing})
            self.capture_last = capture
            self.local_times.append(capture)
            window = self.profile["fps"]["window_seconds"] * 1e9
            while len(self.local_times) > 2 and self.local_times[1] <= capture - window:
                self.local_times.popleft()
            if len(self.local_times) > 1 and capture - self.local_times[0] >= window:
                fps = (len(self.local_times) - 1) * 1e9 / (capture - self.local_times[0])
                self.min_local_fps = fps if self.min_local_fps is None else min(self.min_local_fps, fps)
        if pts is not None:
            self.pts_count += 1
            if self.pts_first is None:
                self.pts_first = pts
            if self.pts_last is not None and pts <= self.pts_last:
                self.pts_invalid = True
            self.pts_last = pts

    def consume(self, frame: FrameRecord):
        pixels = frame.pixels
        if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3 or not pixels.size:
            raise ValueError("Analyzer requires nonempty BGR uint8 frames")
        if self.last_frame_index is not None and frame.frame_index <= self.last_frame_index:
            raise ValueError("frame_index must increase")
        if self.previous is not None and pixels.shape != self.previous.pixels.shape:
            raise ValueError("frame dimensions changed during analysis")
        self.last_frame_index = frame.frame_index
        for event in self.pending:
            event["evidence"].append(self._save(frame, "after"))
        self.pending.clear()
        self._timestamps(frame)
        self.count += 1
        self.decode_bad += frame.decode_status != "ok"
        roi = self.profile["roi"]
        if roi:
            x, y, width, height = roi
            if x + width > pixels.shape[1] or y + height > pixels.shape[0]:
                raise ValueError("ROI extends beyond the frame")
            pixels = pixels[y:y + height, x:x + width]
        gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(pixels, cv2.COLOR_BGR2HSV)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        self.min_sharpness = sharpness if self.min_sharpness is None else min(self.min_sharpness, sharpness)
        self._candidate("blur", sharpness < self.profile["blur"]["min_laplacian_variance"], frame, {"laplacian_variance": sharpness})
        green = self.profile["green"]
        mask = cv2.inRange(hsv, np.array([green["hue_min"], green["min_saturation"], green["min_value"]], dtype=np.uint8),
                           np.array([green["hue_max"], 255, 255], dtype=np.uint8))
        ratio = float(np.count_nonzero(mask) / mask.size)
        self.max_green_ratio = max(self.max_green_ratio, ratio)
        green_metrics = {"green_ratio": ratio}
        if ratio >= green["min_ratio"]:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boxes = sorted((cv2.boundingRect(contour) for contour in contours), key=lambda box: box[2] * box[3], reverse=True)[:10]
            ox, oy = (roi[:2] if roi else (0, 0))
            green_metrics["regions"] = [[x + ox, y + oy, w, h] for x, y, w, h in boxes]
        self._candidate("green", ratio >= green["min_ratio"], frame, green_metrics)
        rgb = pixels.mean(axis=(0, 1))[::-1]
        self.rgb_sum += rgb
        self.hsv_sum += hsv.mean(axis=(0, 1))
        self.ycrcb_sum += cv2.cvtColor(pixels, cv2.COLOR_BGR2YCrCb).mean(axis=(0, 1))
        reference = self.profile["color"]["reference_rgb"]
        if reference is not None:
            distance = float(np.linalg.norm(rgb - reference))
            self.max_color_distance = max(self.max_color_distance, distance)
            self._candidate("color", distance > self.profile["color"]["max_rgb_distance"], frame, {"mean_rgb_euclidean_distance": distance})
        if self.previous_gray is not None:
            difference = float(cv2.absdiff(gray, self.previous_gray).mean())
            if difference <= self.profile["freeze"]["difference_threshold"]:
                if self.freeze_start is None:
                    self.freeze_start = (self.previous.frame_index, self._time(self.previous), self.count - 2)
                elapsed = (self._time(frame) - self.freeze_start[1]) / 1e9 if self._time(frame) is not None and self.freeze_start[1] is not None else (self.count - 1 - self.freeze_start[2]) / self.spec.expected_fps
                if elapsed >= self.profile["freeze"]["min_seconds"]:
                    if "freeze" not in self.active:
                        self.active["freeze"] = self._event("freeze", frame, {"duration_seconds": elapsed, "mean_absolute_difference": difference,
                            "duration_source": "timestamps" if self._time(frame) is not None and self.freeze_start[1] is not None else "expected_fps_estimate"}, self.freeze_start)
                    event = self.active["freeze"]
                    if event:
                        event["end_frame"] = frame.frame_index
                        event["end_time_ns"] = self._time(frame)
                        event["metrics"]["duration_seconds"] = elapsed
            else:
                self.freeze_start = None
                self.active.pop("freeze", None)
            ghost = self.profile["ghost"]
            return_difference = float(cv2.absdiff(gray, self.older_gray).mean()) if self.older_gray is not None else None
            candidate = (return_difference is not None and difference > ghost["difference_threshold"]
                         and self.previous_difference is not None and self.previous_difference > ghost["difference_threshold"]
                         and return_difference <= ghost["return_ratio"] * min(difference, self.previous_difference))
            self._candidate("ghost", candidate, frame, {"adjacent_difference": difference, "two_frame_difference": return_difference,
                            "method": "temporal_return_candidate"})
            self.previous_difference = difference
        self.older_gray = self.previous_gray
        self.previous_gray = gray
        self.previous = frame

    def finish(self):
        checks = []

        def add(name, status, metrics, message, confidence="medium"):
            checks.append({"name": name, "status": status, "metrics": metrics, "thresholds": deepcopy(self.profile.get(name, {})),
                           "message": message, "version": VERSION, "confidence": confidence})

        valid_capture = self.count >= 2 and self.capture_count == self.count and not self.capture_invalid
        capture_fps = ((self.count - 1) * 1e9 / (self.capture_last - self.capture_first)) if valid_capture else None
        valid_pts = self.count >= 2 and self.pts_count == self.count and not self.pts_invalid
        pts_fps = ((self.count - 1) * 1e9 / (self.pts_last - self.pts_first)) if valid_pts else None
        fps_metrics = {"capture_fps": capture_fps, "pts_fps": pts_fps, "timestamp_source": "capture" if valid_capture else "container_pts" if valid_pts else "none",
                       "min_local_fps": self.min_local_fps, "gap_count": self.gaps, "estimated_dropped_frames": self.drops,
                       "expected_fps": self.spec.expected_fps, "frame_count": self.count}
        if self.capture_invalid or (self.capture_count and self.capture_count != self.count):
            add("fps", "ERROR", fps_metrics, "Capture timestamps must be complete and strictly increasing", "high")
        elif not valid_capture:
            add("fps", "UNVERIFIABLE", fps_metrics, "Capture FPS requires at least two trusted per-frame capture timestamps; PTS is descriptive only", "high")
        else:
            tolerance = self.profile["fps"]["tolerance_ratio"]
            bad = abs(capture_fps / self.spec.expected_fps - 1) > tolerance or self.gaps > 0 or (self.min_local_fps is not None and self.min_local_fps < self.spec.expected_fps * (1 - tolerance))
            add("fps", "FAIL" if bad else "PASS", fps_metrics, "Capture timestamp rate and gaps evaluated", "high")
        add("decode", "FAIL" if self.decode_bad else "PASS" if self.count else "ERROR", {"valid_frames": self.count - self.decode_bad, "bad_frames": self.decode_bad}, "Frame decode status; reader diagnostics are evaluated separately", "high")
        for name, metrics in (("freeze", {}), ("green", {"max_green_ratio": self.max_green_ratio}),
                              ("ghost", {"method": "temporal_return_candidate", "limitation": "flags alternating temporal returns; does not prove physical ghosting"}),
                              ("blur", {"min_laplacian_variance": self.min_sharpness}),
                              ("color", {"mean_rgb": (self.rgb_sum / self.count).tolist() if self.count else None,
                                         "mean_hsv": (self.hsv_sum / self.count).tolist() if self.count else None,
                                         "mean_ycrcb": (self.ycrcb_sum / self.count).tolist() if self.count else None,
                                         "max_mean_rgb_euclidean_distance": self.max_color_distance})):
            metrics["event_count"] = self.event_counts[name]
            metrics["stored_events"] = sum(event["detector"] == name for event in self.events)
            if self.count == 0 or (name in ("freeze", "ghost") and self.count < (3 if name == "ghost" else 2)):
                status, message = "UNVERIFIABLE", "Insufficient frames"
            elif name == "color" and self.profile["color"]["reference_rgb"] is None:
                status, message = "UNVERIFIABLE", "Descriptive color statistics only; no reference RGB or color accuracy claim"
            elif self.event_counts[name]:
                status, message = self.candidate_status, "Candidate anomaly; inspect evidence and calibrate scene thresholds"
            else:
                status, message = "PASS", "No candidate exceeded configured thresholds; not a guarantee of subjective quality"
            add(name, status, metrics, message, "low" if status == "SUSPECT" or name == "ghost" else "medium")
        return {"checks": checks, "events": self.events, "frame_count": self.count}
