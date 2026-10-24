"""Generate reproducible synthetic QA inputs; no V9 recordings are implied."""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import yaml

FPS = 30
RESOLUTIONS = ((1920, 1080), (1280, 960), (640, 360))


def normal_frames(width: int, height: int, count: int = 48) -> list[np.ndarray]:
    """Moving high-frequency grayscale texture with repeatable random seed."""
    rng = np.random.default_rng(20261023)
    texture = rng.integers(32, 224, (height, width), dtype=np.uint8)
    return [cv2.cvtColor(np.roll(texture, index * 3, axis=1), cv2.COLOR_GRAY2BGR)
            for index in range(count)]


def timestamps(path: Path, count: int, dropped: bool = False) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_index", "timestamp_ns"])
        for index in range(count):
            writer.writerow([index, round((index + (5 if dropped and index >= 24 else 0)) * 1e9 / FPS)])


def encode(path: Path, frames: list[np.ndarray], codec: str) -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required to generate the compressed matrix")
    height, width = frames[0].shape[:2]
    options = {
        "h264": ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-f", "h264"],
        "h265": ["-c:v", "libx265", "-preset", "ultrafast", "-x265-params", "pools=1:frame-threads=1:log-level=error", "-pix_fmt", "yuv420p", "-f", "hevc"],
        "mjpg": ["-c:v", "mjpeg", "-q:v", "2", "-pix_fmt", "yuvj420p", "-f", "mjpeg"],
    }
    command = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "rawvideo", "-pixel_format", "bgr24",
               "-video_size", f"{width}x{height}", "-framerate", str(FPS), "-i", "pipe:0", "-an", *options[codec], str(path)]
    result = subprocess.run(command, input=b"".join(frame.tobytes() for frame in frames), capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(f"Unable to encode {codec}; FFmpeg must include its encoder: {result.stderr.decode(errors='replace')}")


def generate(output: Path, matrix: bool = False) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    files = []
    if matrix:
        for width, height in RESOLUTIONS:
            frames = normal_frames(width, height, 3)
            for codec in ("raw", "h264", "h265", "mjpg"):
                name = f"{codec}_{width}x{height}"
                path = output / f"{name}.{codec}"
                item = {"file_id": name, "path": path.name, "codec": codec, "width": width, "height": height,
                        "capture_timestamps": f"{name}.csv", "scenario": "synthetic codec/resolution smoke"}
                if codec == "raw":
                    with path.open("wb") as handle:
                        for frame in frames:
                            i420 = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420).ravel()
                            y_size = width * height
                            uv_size = y_size // 4
                            uv = np.column_stack((i420[y_size:y_size + uv_size], i420[y_size + uv_size:])).ravel()
                            handle.write(i420[:y_size].tobytes() + uv.tobytes())
                    item["pixel_format"] = "nv12"
                    item["stride_bytes"] = width
                else:
                    encode(path, frames, codec)
                timestamps(output / item["capture_timestamps"], len(frames))
                files.append(item)
    else:
        (output / "color_profile.yaml").write_text(yaml.safe_dump({
            "mode": "generic",
            "color": {"reference_rgb": [128, 128, 128], "max_rgb_distance": 40},
        }), encoding="utf-8")
        base = normal_frames(64, 48)
        for case in ("normal", "freeze", "dropped_frames", "green", "blur", "color", "ghost"):
            frames = [frame.copy() for frame in base]
            for index in range(8, 38):
                if case == "freeze":
                    frames[index] = base[7].copy()
                elif case == "green":
                    frames[index][8:40, 8:56] = (0, 255, 0)
                elif case == "blur":
                    frames[index] = cv2.GaussianBlur(frames[index], (21, 21), 6)
                elif case == "color":
                    frames[index][:, :, 2] = 255
                    frames[index][:, :, :2] //= 3
                elif case == "ghost":
                    # Alternating return to an earlier frame creates a temporal echo.
                    frames[index] = base[7 if index % 2 == 0 else 8].copy()
            path = output / f"{case}.raw"
            with path.open("wb") as handle:
                for frame in frames:
                    handle.write(frame.tobytes())
            timestamps(output / f"{case}.csv", len(frames), case == "dropped_frames")
            files.append({"file_id": case, "path": path.name, "codec": "raw", "width": 64, "height": 48,
                          "pixel_format": "bgr24", "capture_timestamps": f"{case}.csv", "scenario": f"synthetic {case}",
                          **({"profile": "color_profile.yaml"} if case == "color" else {})})
    manifest = output / "manifest.yaml"
    manifest.write_text(yaml.safe_dump({"project": "synthetic_v1_golden", "expected_fps": FPS, "files": files}, sort_keys=False), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--matrix", action="store_true", help="Generate the 12 format/resolution smoke combinations")
    args = parser.parse_args()
    try:
        print(generate(args.output, args.matrix))
    except (OSError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
