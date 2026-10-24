from copy import deepcopy

import cv2
import numpy as np
import pytest

from videoqa.analyzers import DEFAULT_PROFILE, Analyzer, load_profile, validate_profile
from videoqa.schemas import FileSpec, FrameRecord


def analyze(tmp_path, images, profile=None, timestamps=None, pts=False):
    spec = FileSpec(tmp_path / "input.raw", "raw", expected_fps=10)
    analyzer = Analyzer(spec, profile or DEFAULT_PROFILE, tmp_path / "analysis")
    for index, image in enumerate(images):
        stamp = timestamps[index] if timestamps is not None else None
        analyzer.consume(FrameRecord("test", index, image, width=image.shape[1], height=image.shape[0],
                                     pts_ns=stamp if pts else None,
                                     capture_timestamp_ns=None if pts else stamp))
    return analyzer.finish()


def check(result, name):
    return next(item for item in result["checks"] if item["name"] == name)


def texture():
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)


def test_missing_capture_never_claims_capture_fps(tmp_path):
    result = analyze(tmp_path, [texture()] * 3, timestamps=[0, 100_000_000, 200_000_000], pts=True)
    fps = check(result, "fps")
    assert fps["status"] == "UNVERIFIABLE"
    assert fps["metrics"]["pts_fps"] == 10
    assert fps["metrics"]["capture_fps"] is None
    assert check(result, "color")["status"] == "UNVERIFIABLE"


def test_capture_rate_and_drops(tmp_path):
    images = [texture()] * 4
    result = analyze(tmp_path, images, timestamps=[0, 100_000_000, 300_000_000, 400_000_000])
    fps = check(result, "fps")
    assert fps["status"] == "FAIL"
    assert fps["metrics"]["estimated_dropped_frames"] == 1
    assert fps["metrics"]["gap_count"] == 1
    event = next(item for item in result["events"] if item["detector"] == "fps")
    assert event["start_frame"] == 2
    assert event["start_time_ns"] == 300_000_000


def test_trusted_capture_passes_and_invalid_fails(tmp_path):
    result = analyze(tmp_path, [texture()] * 3, timestamps=[0, 100_000_000, 200_000_000])
    assert check(result, "fps")["status"] == "PASS"
    invalid = analyze(tmp_path, [texture()] * 3, timestamps=[0, 0, 200_000_000])
    assert check(invalid, "fps")["status"] == "ERROR"


def test_freeze_duration_evidence_and_candidate(tmp_path):
    result = analyze(tmp_path, [texture()] * 15)
    assert check(result, "freeze")["status"] == "SUSPECT"
    event = next(item for item in result["events"] if item["detector"] == "freeze")
    assert event["start_frame"] == 0
    assert event["end_frame"] == 14
    assert event["metrics"]["duration_seconds"] == 1.4
    assert event["metrics"]["duration_source"] == "expected_fps_estimate"
    assert len(event["evidence"]) == 3
    for path in event["evidence"]:
        assert (tmp_path / "analysis" / path).is_file()


def test_green_blur_color_and_reference_status(tmp_path):
    green = np.full((48, 64, 3), (0, 255, 0), dtype=np.uint8)
    profile = deepcopy(DEFAULT_PROFILE)
    profile.update(mode="reference", calibrated=True)
    profile["color"]["reference_rgb"] = [128, 128, 128]
    result = analyze(tmp_path, [green] * 3, profile)
    for name in ("green", "blur", "color"):
        assert check(result, name)["status"] == "FAIL"
    assert check(result, "green")["metrics"]["max_green_ratio"] == 1
    event = next(item for item in result["events"] if item["detector"] == "green")
    assert event["metrics"]["regions"] == [[0, 0, 64, 48]]
    profile["calibrated"] = False
    assert check(analyze(tmp_path, [green] * 3, profile), "green")["status"] == "SUSPECT"


def test_ghost_is_separate_temporal_candidate(tmp_path):
    a = texture()
    b = 255 - a
    result = analyze(tmp_path, [a, b, a, b])
    assert check(result, "ghost")["status"] == "SUSPECT"
    assert check(result, "freeze")["status"] == "PASS"
    event = next(item for item in result["events"] if item["detector"] == "ghost")
    assert event["metrics"]["method"] == "temporal_return_candidate"


def test_roi_applies_color_statistics_and_bounds(tmp_path):
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    image[:10, :10] = (0, 0, 255)
    profile = deepcopy(DEFAULT_PROFILE)
    profile["roi"] = [0, 0, 10, 10]
    result = analyze(tmp_path, [image], profile)
    assert check(result, "color")["metrics"]["mean_rgb"] == [255, 0, 0]
    profile["roi"] = [60, 40, 10, 10]
    with pytest.raises(ValueError, match="ROI"):
        analyze(tmp_path, [image], profile)


def test_event_storage_bounded_and_counts_preserved(tmp_path):
    a = texture()
    b = cv2.GaussianBlur(a, (21, 21), 0)
    profile = deepcopy(DEFAULT_PROFILE)
    profile["max_events"] = 2
    result = analyze(tmp_path, [a, b] * 10, profile)
    blur = check(result, "blur")
    assert blur["metrics"]["event_count"] == 10
    assert blur["metrics"]["stored_events"] == 2
    assert len([e for e in result["events"] if e["detector"] == "blur"]) == 2


@pytest.mark.parametrize("profile", [None, [], {"unknown": 1}, {"blur": {"typo": 2}},
    {"fps": {"tolerance_ratio": float("nan")}}, {"freeze": {"min_seconds": -1}},
    {"green": {"hue_min": 100, "hue_max": 50}}, {"color": {"reference_rgb": [300, 0, 0]}},
    {"max_events": True}, {"roi": [0, 0, -1, 1]}, {"calibrated": "yes"}])
def test_reject_invalid_profiles(profile):
    with pytest.raises(ValueError):
        validate_profile(profile)


def test_load_profile_merges_without_mutation(tmp_path):
    path = tmp_path / "profile.yaml"
    path.write_text("freeze:\n  min_seconds: 0.2\n")
    loaded = load_profile(path)
    assert loaded["freeze"]["min_seconds"] == 0.2
    assert DEFAULT_PROFILE["freeze"]["min_seconds"] == 1
    assert loaded["freeze"]["difference_threshold"] == 0.5


def test_zero_frames_explicit(tmp_path):
    result = analyze(tmp_path, [])
    assert check(result, "decode")["status"] == "ERROR"
    assert check(result, "fps")["status"] == "UNVERIFIABLE"
