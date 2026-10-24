"""Exercise real FFmpeg decoders and the complete CLI/report pipeline."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest
import yaml

from scripts.generate_golden import generate
from videoqa.schemas import FileSpec


@pytest.fixture(scope="module")
def matrix(tmp_path_factory):
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("FFmpeg and ffprobe required for codec integration tests")
    root = tmp_path_factory.mktemp("matrix")
    generate(root, matrix=True)
    return root


@pytest.mark.parametrize("codec", ["raw", "h264", "h265", "mjpg"])
@pytest.mark.parametrize("width,height", [(1920, 1080), (1280, 960), (640, 360)])
def test_planned_codec_resolution_matrix(matrix, codec, width, height):
    from videoqa.readers import open_reader

    name = f"{codec}_{width}x{height}"
    spec = FileSpec(path=matrix / f"{name}.{codec}", codec=codec, width=width, height=height,
                    capture_timestamps=matrix / f"{name}.csv",
                    pixel_format="nv12" if codec == "raw" else None,
                    stride_bytes=width if codec == "raw" else None)
    reader = open_reader(spec)
    try:
        count = 0
        for frame in reader:
            assert frame.pixels.shape == (height, width, 3)
            assert frame.frame_index == count
            assert frame.capture_timestamp_ns == round(count * 1e9 / 30)
            count += 1
        assert count == 3
        assert not reader.diagnostics
    finally:
        reader.close()


def run_cli(*args):
    return subprocess.run([sys.executable, "-m", "videoqa", *map(str, args)], capture_output=True, text=True, check=False)


def test_golden_batch_reports_and_evidence(tmp_path):
    manifest = generate(tmp_path / "dataset")
    output = tmp_path / "reports"
    result = run_cli("batch", "--manifest", manifest, "--output", output,
                     "--suspect", "warn", "--unverifiable", "warn")
    assert result.returncode == 1, result.stdout + result.stderr  # trusted capture gaps fail FPS
    for name in ("report.json", "report.csv", "report.html", "junit.xml"):
        assert (output / name).is_file()
    report = json.loads((output / "report.json").read_text())
    results = report["results"]
    assert len(results) == 7
    by_name = {item["file_id"]: item for item in results}
    assert by_name["normal"]["frame_count"] == 48
    for name, detector in (("freeze", "freeze"), ("green", "green"), ("blur", "blur"), ("ghost", "ghost"), ("color", "color")):
        assert any(event["detector"] == detector for event in by_name[name]["events"]), name
    evidence = [entry for item in results for event in item["events"] for entry in event["evidence"]]
    assert evidence
    assert all((output / entry).is_file() for entry in evidence)
    ET.parse(output / "junit.xml")


def test_batch_continues_after_invalid_file(tmp_path):
    manifest = generate(tmp_path / "dataset")
    data = yaml.safe_load(manifest.read_text())
    data["files"].insert(0, {"file_id": "missing", "path": "missing.raw", "codec": "raw",
                            "width": 64, "height": 48, "pixel_format": "bgr24"})
    manifest.write_text(yaml.safe_dump(data))
    output = tmp_path / "reports"
    result = run_cli("batch", "--manifest", manifest, "--output", output,
                     "--suspect", "warn", "--unverifiable", "warn")
    assert result.returncode == 2, result.stdout + result.stderr
    results = json.loads((output / "report.json").read_text())["results"]
    assert len(results) == 8
    assert results[0]["status"] == "ERROR"
    assert next(item for item in results if item["file_id"] == "normal")["frame_count"] == 48


def test_truncated_compressed_input_reports_decode_error(matrix, tmp_path):
    original = matrix / "h264_640x360.h264"
    damaged = tmp_path / "damaged.h264"
    content = original.read_bytes()
    damaged.write_bytes(content[:len(content) // 2])
    output = tmp_path / "reports"
    result = run_cli("analyze", "--input", damaged, "--codec", "h264", "--output", output,
                     "--suspect", "warn", "--unverifiable", "warn")
    report = json.loads((output / "report.json").read_text())
    checks = report["results"][0]["checks"]
    assert result.returncode in (1, 2), result.stdout + result.stderr
    assert any(check["name"] == "decode" and check["status"] in ("FAIL", "ERROR") for check in checks)
