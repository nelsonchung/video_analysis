"""Manifest and single-file entry points for the CPU baseline."""
import argparse
import re
import resource
import subprocess
import sys
import time
from dataclasses import fields
from pathlib import Path

import yaml

from . import __version__
from .analyzers import Analyzer, load_profile
from .readers import open_reader
from .reports import write_reports
from .schemas import FileSpec


def aggregate_status(checks):
    statuses = {check["status"] for check in checks}
    return next((s for s in ("ERROR", "FAIL", "SUSPECT", "UNVERIFIABLE") if s in statuses), "PASS")


def exit_code(results, policy):
    statuses = {c["status"] for result in results for c in result["checks"]}
    if "ERROR" in statuses:
        return 2
    if "FAIL" in statuses or any(policy[s.lower()] == "fail" for s in statuses & {"SUSPECT", "UNVERIFIABLE"}):
        return 1
    return 0


def error_check(message):
    return {"name": "decode", "status": "ERROR", "message": str(message), "metrics": {},
            "thresholds": {}, "version": __version__, "confidence": "high"}


def analyze_file(spec, profile, output_dir):
    start = time.perf_counter()
    reader = None
    analyzer = Analyzer(spec, profile, output_dir)
    result = {"file_id": spec.file_id, "path": str(spec.path), "codec": spec.codec,
              "scenario": spec.scenario, "metadata": {}, "diagnostics": [], "events": [], "checks": [], "frame_count": 0}
    problem = None
    try:
        reader = open_reader(spec)
        for frame in reader:
            analyzer.consume(frame)
    except (ValueError, OSError, RuntimeError, TypeError, subprocess.SubprocessError) as exc:
        problem = exc
    finally:
        if reader is not None:
            reader.close()
            result["metadata"] = reader.metadata
            result["diagnostics"] = list(reader.diagnostics)
    result.update(analyzer.finish())
    # Merge reader diagnostics into the analyzer decode check, keeping one check per module.
    result["checks"] = [check for check in result["checks"] if check["name"] != "decode"]
    if problem is not None:
        result["checks"].append(error_check(problem))
    elif result["frame_count"] == 0:
        result["checks"].append(error_check("No valid frames decoded"))
    else:
        diagnostics = result["diagnostics"]
        result["checks"].append({"name": "decode", "status": "FAIL" if diagnostics else "PASS",
                                 "message": "; ".join(diagnostics) or "All decoded frames valid",
                                 "metrics": {"valid_frames": result["frame_count"], "diagnostic_count": len(diagnostics)},
                                 "thresholds": {"diagnostic_count": 0}, "version": __version__, "confidence": "high"})
    result["status"] = aggregate_status(result["checks"])
    elapsed = time.perf_counter() - start
    # getrusage peak is process-wide (includes earlier batch files), not per-file allocation.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result["performance"] = {"elapsed_seconds": elapsed, "frames_per_second": result["frame_count"] / elapsed,
                             "process_peak_rss_bytes": rss if sys.platform == "darwin" else rss * 1024,
                             "backend": "cpu", "memory_scope": "process high-water mark"}
    return result


def resolve_profile(value, base):
    if value is None:
        return load_profile()
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return load_profile(path)


def load_manifest(path):
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict) or not isinstance(data.get("files"), list) or not data["files"]:
        raise ValueError("Manifest must contain a nonempty files list")
    unknown = set(data) - {"project", "expected_fps", "profile", "files", "scenario"}
    if unknown:
        raise ValueError(f"Unknown manifest keys: {sorted(unknown)}")
    entries = []
    allowed = {f.name for f in fields(FileSpec)}
    for index, item in enumerate(data["files"]):
        if not isinstance(item, dict) or not {"path", "codec"} <= item.keys():
            raise ValueError(f"files[{index}] requires path and codec")
        unknown = set(item) - allowed
        if unknown:
            raise ValueError(f"Unknown file keys: {sorted(unknown)}")
        config = {"expected_fps": data.get("expected_fps", 30), "scenario": data.get("scenario", "unspecified"), **item}
        for key in ("path", "capture_timestamps"):
            if config.get(key) is not None:
                config[key] = (path.parent / config[key]).resolve()
        spec = FileSpec(**config)
        profile = resolve_profile(item.get("profile", data.get("profile")), path.parent)
        entries.append((spec, profile))
    return entries


def parser():
    root = argparse.ArgumentParser(description="V9 offline video quality analyzer (CPU)")
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)
    for command in ("analyze", "batch"):
        sub = commands.add_parser(command)
        sub.add_argument("--output", type=Path, default=Path("reports/run"))
        sub.add_argument("--suspect", choices=("fail", "warn"), default="fail")
        sub.add_argument("--unverifiable", choices=("fail", "warn"), default="fail")
        if command == "batch":
            sub.add_argument("--manifest", type=Path, required=True)
            continue
        sub.add_argument("--input", type=Path, required=True)
        sub.add_argument("--codec", choices=("raw", "h264", "h265", "mjpg"), required=True)
        sub.add_argument("--profile", type=Path)
        sub.add_argument("--expected-fps", type=float, default=30)
        sub.add_argument("--capture-timestamps", type=Path)
        sub.add_argument("--pixel-format")
        sub.add_argument("--packing", default="unpacked")
        sub.add_argument("--file-id", default="")
        sub.add_argument("--scenario", default="unspecified")
        for option in ("width", "height", "stride-bytes", "frame-size-bytes", "header-bytes", "bit-depth"):
            sub.add_argument(f"--{option}", type=int, default={"header-bytes": 0, "bit-depth": 8}.get(option))
        sub.add_argument("--plane-offsets", type=int, nargs="+")
        sub.add_argument("--plane-strides", type=int, nargs="+")
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    policy = {"suspect": args.suspect, "unverifiable": args.unverifiable}
    try:
        if args.command == "batch":
            entries = load_manifest(args.manifest.resolve())
        else:
            values = {f.name: getattr(args, f.name) for f in fields(FileSpec) if hasattr(args, f.name)}
            values["path"] = args.input.resolve()
            spec = FileSpec(**values)
            entries = [(spec, resolve_profile(args.profile, Path.cwd()))]
        args.output.mkdir(parents=True, exist_ok=True)
        results = []
        for index, (spec, profile) in enumerate(entries):
            directory = f"{index:03d}_" + (re.sub(r"[^A-Za-z0-9_.-]", "_", spec.file_id)[:80] or "file")
            file_output = args.output / directory
            file_output.mkdir(exist_ok=True)
            result = analyze_file(spec, profile, file_output)
            for event in result["events"]:
                event["evidence"] = [f"{directory}/{path}" for path in event.get("evidence", [])]
            results.append(result)
            print(f"{spec.file_id}: {result['status']} ({result['frame_count']} frames)")
        write_reports(results, args.output, policy)
        print(f"Reports: {args.output.resolve()}")
        return exit_code(results, policy)
    except (ValueError, OSError, RuntimeError, TypeError, subprocess.SubprocessError, yaml.YAMLError) as exc:
        print(f"videoqa: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
