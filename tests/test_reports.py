import csv
import json
import xml.etree.ElementTree as ET

import pytest

from videoqa.reports import write_reports


def sample():
    return {
        "file_id": "camera<&", "path": "video<&.raw", "codec": "raw", "status": "SUSPECT",
        "frame_count": 12, "checks": [
            {"name": "freeze", "status": "SUSPECT", "message": "Review <script>alert(1)</script>",
             "metrics": {"duration_seconds": 2}, "thresholds": {"seconds": 1}, "confidence": "low"},
            {"name": "fps", "status": "UNVERIFIABLE", "message": "No capture timestamps"},
            {"name": "decode", "status": "PASS"},
        ], "events": [{"detector": "freeze", "start_frame": 3, "end_frame": 7,
                       "evidence": ["camera/frame 3.png", "javascript:alert(1)", "../outside.png"]}],
        "metadata": {"timestamp_source": "none"}, "diagnostics": [], "performance": {},
    }


def test_all_reports_preserve_metrics_policy_and_escape_content(tmp_path):
    write_reports([sample()], tmp_path)
    data = json.loads((tmp_path / "report.json").read_text())
    assert data["policy"] == {"suspect": "fail", "unverifiable": "fail"}
    assert data["results"][0]["events"][0]["start_frame"] == 3
    with (tmp_path / "report.csv").open(newline="") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 3
    assert json.loads(rows[0]["metrics"])["duration_seconds"] == 2
    assert json.loads(rows[0]["events"])[0]["start_frame"] == 3
    page = (tmp_path / "report.html").read_text()
    assert "<script>" not in page
    assert "&lt;script&gt;" in page
    assert 'src="camera/frame%203.png"' in page
    assert 'href="javascript:' not in page
    assert 'href="../' not in page
    assert "timestamp_source" in page
    xml = ET.parse(tmp_path / "junit.xml").getroot()
    assert xml.attrib["tests"] == "3"
    assert xml.attrib["failures"] == "2"
    assert xml.attrib["skipped"] == "0"


def test_warn_is_explicit_junit_skip(tmp_path):
    write_reports([sample()], tmp_path, {"suspect": "warn", "unverifiable": "warn"})
    xml = ET.parse(tmp_path / "junit.xml").getroot()
    assert xml.attrib["failures"] == "0"
    assert xml.attrib["skipped"] == "2"
    assert json.loads((tmp_path / "report.json").read_text())["results"][0]["status"] == "SUSPECT"


def test_file_error_not_lost_after_completed_checks(tmp_path):
    result = sample()
    result.update(status="ERROR", checks=[{"name": "fps", "status": "PASS"}], diagnostics=["truncated frame"])
    write_reports([result], tmp_path)
    xml = ET.parse(tmp_path / "junit.xml").getroot()
    assert xml.attrib["tests"] == "2"
    assert xml.attrib["errors"] == "1"
    assert xml.find(".//error").attrib["message"] == "truncated frame"


def test_error_fail_and_pass_mapping(tmp_path):
    result = sample()
    result["checks"] = [{"name": status, "status": status} for status in ["ERROR", "FAIL", "PASS"]]
    write_reports([result], tmp_path)
    xml = ET.parse(tmp_path / "junit.xml").getroot()
    assert xml.attrib["errors"] == "1"
    assert xml.attrib["failures"] == "1"


@pytest.mark.parametrize("policy", [{"suspect": "pass"}, {"other": "warn"}])
def test_invalid_policy_rejected(tmp_path, policy):
    with pytest.raises(ValueError):
        write_reports([], tmp_path, policy)


def test_nonfinite_metrics_rejected_before_writing(tmp_path):
    result = sample()
    result["checks"][0]["metrics"]["score"] = float("nan")
    with pytest.raises(ValueError):
        write_reports([result], tmp_path)
    assert not list(tmp_path.iterdir())
