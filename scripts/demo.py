"""Generate and verify deterministic PASS / FAIL / ERROR demonstration files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from scripts.generate_golden import encode, generate, normal_frames, timestamps
from videoqa.cli import main as analyze

CHECKS = ("fps", "decode", "freeze", "green", "ghost", "blur", "color")


def generate_demo(output: Path) -> Path:
    """Reuse the golden fixtures with a profile calibrated only for these fixtures."""
    output = Path(output).resolve()
    manifest = generate(output)
    files = yaml.safe_load(manifest.read_text(encoding="utf-8"))["files"]
    profile = {
        "mode": "reference", "calibrated": True,
        "color": {"reference_rgb": [128, 128, 128], "max_rgb_distance": 40},
    }
    (output / "synthetic_reference.yaml").write_text(yaml.safe_dump(profile), encoding="utf-8")
    for item in files:
        item["profile"] = "synthetic_reference.yaml"
    frames = normal_frames(64, 48)
    for codec in ("h264", "h265", "mjpg"):
        name = f"normal_{codec}"
        encode(output / f"{name}.{codec}", frames, codec)
        timestamps(output / f"{name}.csv", len(frames))
        files.append({
            "file_id": name, "path": f"{name}.{codec}", "codec": codec,
            "width": 64, "height": 48, "capture_timestamps": f"{name}.csv",
            "profile": "synthetic_reference.yaml", "scenario": "synthetic moving neutral texture",
        })
    # A valid path with an invalid frame length deterministically exercises ERROR.
    (output / "truncated.raw").write_bytes(bytes(64 * 48 * 3 - 1))
    files.append({"file_id": "truncated", "path": "truncated.raw", "codec": "raw",
                  "width": 64, "height": 48, "pixel_format": "bgr24",
                  "profile": "synthetic_reference.yaml", "scenario": "synthetic partial RAW frame"})
    positive = {"normal", "normal_h264", "normal_h265", "normal_mjpg"}
    groups = {
        "pass": [item for item in files if item["file_id"] in positive],
        "fail": [item for item in files if item["file_id"] not in positive | {"truncated"}],
        "error": [item for item in files if item["file_id"] == "truncated"],
    }
    expectations = {"groups": {}}
    targets = {"freeze": "freeze", "dropped_frames": "fps", "green": "green",
               "blur": "blur", "color": "color", "ghost": "ghost"}
    for group, entries in groups.items():
        data = {"project": f"demo_{group}", "expected_fps": 30, "files": entries}
        (output / f"{group}.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        expected_files = {}
        for item in entries:
            name = item["file_id"]
            checks = dict.fromkeys(CHECKS, "PASS") if group == "pass" else {
                "decode" if group == "error" else targets[name]: "ERROR" if group == "error" else "FAIL",
            }
            expected_files[name] = {"status": group.upper(), "checks": checks}
        expectations["groups"][group] = {
            "manifest": f"{group}.yaml", "exit_code": {"pass": 0, "fail": 1, "error": 2}[group],
            "files": expected_files,
        }
    (output / "expectations.json").write_text(json.dumps(expectations, indent=2) + "\n", encoding="utf-8")
    return output / "expectations.json"


def verify_demo(dataset: Path, reports: Path) -> dict:
    """Compare actual reports and exit codes with independent expected outcomes."""
    dataset, reports = Path(dataset).resolve(), Path(reports).resolve()
    expected = json.loads((dataset / "expectations.json").read_text(encoding="utf-8"))
    comparisons = []

    def compare(label, actual, wanted):
        comparisons.append({"item": label, "expected": wanted, "actual": actual, "matched": actual == wanted})

    for group, config in expected["groups"].items():
        directory = reports / group
        code = analyze(["batch", "--manifest", str(dataset / config["manifest"]), "--output", str(directory)])
        compare(f"{group}.exit_code", code, config["exit_code"])
        if not (directory / "report.json").exists():
            compare(f"{group}.report_exists", False, True)
            continue
        results = json.loads((directory / "report.json").read_text(encoding="utf-8"))["results"]
        actual_files = {item["file_id"]: item for item in results}
        compare(f"{group}.files", sorted(actual_files), sorted(config["files"]))
        for name, wanted in config["files"].items():
            result = actual_files.get(name, {})
            compare(f"{name}.status", result.get("status"), wanted["status"])
            actual_checks = {check["name"]: check["status"] for check in result.get("checks", [])}
            for check, status in wanted["checks"].items():
                compare(f"{name}.{check}", actual_checks.get(check), status)
            if group == "fail":
                for detector in wanted["checks"]:
                    events = [event for event in result.get("events", []) if event["detector"] == detector]
                    compare(f"{name}.{detector}.event_exists", bool(events), True)
                    evidence = [path for event in events for path in event.get("evidence", [])]
                    compare(f"{name}.{detector}.evidence_exists",
                            bool(evidence) and all((directory / path).is_file() for path in evidence), True)
    summary = {"status": "PASS" if all(item["matched"] for item in comparisons) else "FAIL",
               "comparisons": comparisons}
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "verification.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("demos/generated"))
    parser.add_argument("--reports", type=Path, default=Path("reports/demo"))
    args = parser.parse_args(argv)
    try:
        generate_demo(args.output)
        summary = verify_demo(args.output, args.reports)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"demo: {exc}\n")
    for item in summary["comparisons"]:
        if not item["matched"]:
            print(f"MISMATCH {item['item']}: expected {item['expected']}, got {item['actual']}")
    print(f"Demo verification: {summary['status']} ({len(summary['comparisons'])} comparisons)")
    print(f"Summary: {(args.reports / 'verification.json').resolve()}")
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
