"""Reader format contracts and loss/corruption diagnostics."""
import shutil
import subprocess

import numpy as np
import pytest

from videoqa.readers import open_reader
from videoqa.schemas import FileSpec


def raw_spec(tmp_path, data, **kwargs):
    path = tmp_path / "sample.raw"
    path.write_bytes(data)
    return FileSpec(path, "raw", width=4, height=2, pixel_format="bgr24", **kwargs)


def test_raw_stride_header_padding_and_capture(tmp_path):
    pixels = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
    data = b"HEAD" + pixels[0].tobytes() + b"PAD!" + pixels[1].tobytes() + b"PAD!" + b"TAIL"
    timestamps = tmp_path / "capture.csv"
    timestamps.write_text("frame_index,timestamp_ns\n0,1234567890\n1,1267901223\n")
    spec = raw_spec(tmp_path, data * 2, stride_bytes=16, header_bytes=4,
                    frame_size_bytes=40, capture_timestamps=timestamps)
    with open_reader(spec) as reader:
        frames = list(reader)
    assert len(frames) == 2
    np.testing.assert_array_equal(frames[0].pixels, pixels)
    assert frames[1].capture_timestamp_ns == 1267901223
    assert frames[0].pts_ns is None


@pytest.mark.parametrize("fmt", ["nv12", "i420", "yuyv422", "gray", "rgb24", "bayer_rggb8"])
def test_raw_formats(tmp_path, fmt):
    sizes = {"nv12": 12, "i420": 12, "yuyv422": 16, "gray": 8, "rgb24": 24, "bayer_rggb8": 8}
    spec = raw_spec(tmp_path, bytes([128]) * sizes[fmt])
    spec.pixel_format = fmt
    with open_reader(spec) as reader:
        frame = next(iter(reader))
        assert frame.pixels.shape == (2, 4, 3)


def test_raw_plane_offsets(tmp_path):
    # Neutral I420 image with gaps between all planes.
    data = bytes([16]) * 8 + b"gap!" + bytes([128]) * 2 + b"gap!" + bytes([128]) * 2
    spec = raw_spec(tmp_path, data, plane_offsets=[0, 12, 18])
    spec.pixel_format = "i420"
    with open_reader(spec) as reader:
        assert np.max(next(iter(reader)).pixels) == 0


@pytest.mark.parametrize("changes,match", [
    ({"pixel_format": None}, "requires"), ({"bit_depth": 10}, "8-bit"),
    ({"stride_bytes": 1}, "stride"), ({"frame_size_bytes": 1}, "cannot contain"),
    ({"plane_offsets": [0, 1]}, "one entry"), ({"packing": "packed"}, "8-bit"),
    ({"plane_offsets": 4}, "must be a list"), ({"plane_strides": 4}, "must be a list"),
    ({"pixel_format": 123}, "must be a string"),
])
def test_raw_rejects_bad_layout(tmp_path, changes, match):
    spec = raw_spec(tmp_path, bytes(24))
    for key, value in changes.items():
        setattr(spec, key, value)
    with pytest.raises(ValueError, match=match):
        open_reader(spec)


def test_raw_truncation(tmp_path):
    with pytest.raises(ValueError, match="multiple"):
        open_reader(raw_spec(tmp_path, bytes(25)))


@pytest.mark.parametrize("rows,match", [
    ("time\n0\n", "timestamp_ns"),
    ("timestamp_ns\n0\n0\n", "strictly increasing"),
    ("timestamp_ns\n0\n", "fewer rows"),
    ("timestamp_ns\n0\n1\n2\n", "more rows"),
    ("frame_index,timestamp_ns\n1,0\n2,1\n", "contiguous"),
])
def test_timestamp_validation(tmp_path, rows, match):
    times = tmp_path / "times.csv"
    times.write_text(rows)
    spec = raw_spec(tmp_path, bytes(48), capture_timestamps=times)
    with pytest.raises(ValueError, match=match), open_reader(spec) as reader:
        list(reader)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg not installed")
@pytest.mark.parametrize("codec,encoder,extension", [
    ("h264", "libx264", "mp4"), ("h265", "libx265", "hevc"), ("mjpg", "mjpeg", "mjpg"),
])
def test_ffmpeg_codecs(tmp_path, codec, encoder, extension):
    path = tmp_path / f"video.{extension}"
    command = ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=64x48:rate=10", "-frames:v", "3", "-c:v", encoder]
    if codec == "h265":
        command += ["-x265-params", "pools=1:frame-threads=1:log-level=error"]
    subprocess.run(command + [str(path)], check=True, capture_output=True)
    with open_reader(FileSpec(path, codec, 64, 48)) as reader:
        frames = list(reader)
        assert not reader.diagnostics
        assert reader.metadata["frame_count"] == 3
    assert len(frames) == 3
    if extension == "mp4":
        assert frames[1].pts_ns == 100_000_000
    else:
        assert all(f.capture_timestamp_ns is None for f in frames)
    assert frames[0].pixels.shape == (48, 64, 3)
    with pytest.raises(ValueError, match="Resolution mismatch"):
        open_reader(FileSpec(path, codec, 32, 24))


def test_ffmpeg_rejects_missing_input(tmp_path):
    with pytest.raises(ValueError, match="ffprobe failed"):
        open_reader(FileSpec(tmp_path / "missing.mp4", "h264"))
