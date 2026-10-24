import json
import shutil

import pytest

from scripts.demo import generate_demo, verify_demo


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg required")
def test_demo_expected_pass_fail_error_and_mismatch_detection(tmp_path):
    dataset = tmp_path / "dataset"
    expectations = generate_demo(dataset)
    summary = verify_demo(dataset, tmp_path / "reports")
    assert summary["status"] == "PASS", [item for item in summary["comparisons"] if not item["matched"]]
    assert len(summary["comparisons"]) >= 60
    data = json.loads(expectations.read_text())
    data["groups"]["pass"]["files"]["normal"]["checks"]["fps"] = "FAIL"
    expectations.write_text(json.dumps(data))
    mismatch = verify_demo(dataset, tmp_path / "reports")
    assert mismatch["status"] == "FAIL"
    assert any(item["item"] == "normal.fps" and not item["matched"] for item in mismatch["comparisons"])
