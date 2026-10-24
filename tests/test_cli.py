import json

import numpy as np
import pytest
import yaml

from videoqa.cli import exit_code, load_manifest, main


def test_policy():
    for status, expected in [("PASS", 0), ("FAIL", 1), ("ERROR", 2), ("SUSPECT", 1), ("UNVERIFIABLE", 1)]:
        assert exit_code([{"checks": [{"status": status}]}], {"suspect": "fail", "unverifiable": "fail"}) == expected
    assert exit_code([{"checks": [{"status": "UNVERIFIABLE"}]}], {"suspect": "warn", "unverifiable": "warn"}) == 0


def test_missing_file_writes_error_report(tmp_path):
    output = tmp_path / "report"
    assert main(["analyze", "--input", str(tmp_path / "missing.h264"), "--codec", "h264", "--output", str(output)]) == 2
    data = json.loads((output / "report.json").read_text())
    assert data["results"][0]["status"] == "ERROR"


def test_batch_continues_after_reader_error(tmp_path):
    (tmp_path / "ok.raw").write_bytes(np.zeros((3, 8, 8, 3), dtype=np.uint8).tobytes())
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(yaml.safe_dump({"files": [
        {"path": "missing.raw", "codec": "raw", "width": 8, "height": 8, "pixel_format": "bgr24"},
        {"path": "ok.raw", "codec": "raw", "width": 8, "height": 8, "pixel_format": "bgr24"}]}))
    output = tmp_path / "out"
    assert main(["batch", "--manifest", str(manifest), "--output", str(output)]) == 2
    results = json.loads((output / "report.json").read_text())["results"]
    assert len(results) == 2
    assert results[1]["frame_count"] == 3


def test_manifest_rejects_typos_and_resolves_relative_paths(tmp_path):
    path = tmp_path / "manifest.yaml"
    path.write_text('files:\n- {path: video.raw, codec: raw, width: 4, height: 4, pixel_format: gray}\n')
    entries = load_manifest(path)
    assert entries[0][0].path == tmp_path / "video.raw"
    path.write_text('files:\n- {path: video.raw, codec: raw, expected_fp: 30}\n')
    with pytest.raises(ValueError, match="Unknown file keys"):
        load_manifest(path)


def test_invalid_manifest_returns_input_error(tmp_path):
    path = tmp_path / "manifest.yaml"
    path.write_text('files: []\n')
    assert main(["batch", "--manifest", str(path)]) == 2


def test_rerun_replaces_evidence(tmp_path):
    import cv2
    video = tmp_path / "input.raw"
    output = tmp_path / "out"
    args = ["analyze", "--input", str(video), "--codec", "raw", "--width", "8", "--height", "8",
            "--pixel-format", "bgr24", "--output", str(output), "--suspect", "warn", "--unverifiable", "warn"]
    for level in (0, 200):
        video.write_bytes(np.full((3, 8, 8, 3), level, dtype=np.uint8).tobytes())
        assert main(args) == 0
        result = json.loads((output / "report.json").read_text())["results"][0]
        event = next(e for e in result["events"] if e["detector"] == "blur")
        image = cv2.imread(str(output / event["evidence"][0]))
        assert abs(float(image.mean()) - level) < 2
