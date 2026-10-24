"""Portable, escaped reports with explicit CI handling of uncertain findings."""
from __future__ import annotations

import csv
import io
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from html import escape
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote, urlsplit

from .schemas import STATUSES, VERSION


def _json(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=indent)


def _policy(policy: dict | None) -> dict[str, str]:
    result = {"suspect": "fail", "unverifiable": "fail"}
    if policy:
        if set(policy) - set(result):
            raise ValueError("CI policy supports only suspect and unverifiable")
        result.update(policy)
    if any(value not in {"fail", "warn"} for value in result.values()):
        raise ValueError("CI policy values must be fail or warn")
    return result


def _checks(result: dict) -> list[dict]:
    checks = list(result.get("checks", []))
    # Preserve a file-level execution error even if earlier checks completed.
    if not checks or (result.get("status") == "ERROR" and not any(
        check.get("status") == "ERROR" for check in checks
    )):
        checks.append({
            "name": "file", "status": result.get("status", "ERROR"),
            "message": "; ".join(str(item) for item in result.get("diagnostics", [])),
        })
    return checks


def _outcome(status: str, policy: dict[str, str]) -> str | None:
    if status == "ERROR":
        return "error"
    if status == "FAIL":
        return "failure"
    if status in {"SUSPECT", "UNVERIFIABLE"}:
        return "failure" if policy[status.lower()] == "fail" else "skipped"
    return None


def _junit(results: list[dict], policy: dict[str, str]) -> str:
    suite = ET.Element("testsuite", name="videoqa", tests="0", failures="0", errors="0", skipped="0")
    properties = ET.SubElement(suite, "properties")
    for key, value in policy.items():
        ET.SubElement(properties, "property", name=f"policy.{key}", value=value)
    counts = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    for result in results:
        for check in _checks(result):
            counts["tests"] += 1
            status = check.get("status", "ERROR")
            case = ET.SubElement(
                suite, "testcase", classname=str(result.get("file_id", "unknown")),
                name=str(check.get("name", "check")),
            )
            details = _json({
                "path": result.get("path"), "status": status,
                "metrics": check.get("metrics", {}), "thresholds": check.get("thresholds", {}),
                "version": check.get("version", VERSION), "confidence": check.get("confidence"),
                "message": check.get("message", ""),
                "metadata": result.get("metadata", {}),
                "events": [event for event in result.get("events", [])
                           if event.get("detector") == check.get("name")],
            })
            outcome = _outcome(status, policy)
            if outcome:
                counts[{"failure": "failures", "error": "errors", "skipped": "skipped"}[outcome]] += 1
                ET.SubElement(case, outcome, type=status, message=str(check.get("message", status))).text = details
            ET.SubElement(case, "system-out").text = details
    for key, value in counts.items():
        suite.set(key, str(value))
    ET.indent(suite, space="  ")
    return '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(suite, encoding="unicode") + "\n"


def _csv(results: list[dict]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=[
        "file_id", "path", "codec", "file_status", "frame_count", "check", "status",
        "message", "metrics", "thresholds", "version", "confidence", "events",
    ])
    writer.writeheader()
    for result in results:
        for check in _checks(result):
            writer.writerow({
                "file_id": result.get("file_id", ""), "path": result.get("path", ""),
                "codec": result.get("codec", ""), "file_status": result.get("status", ""),
                "frame_count": result.get("frame_count", 0), "check": check.get("name", ""),
                "status": check.get("status", "ERROR"), "message": check.get("message", ""),
                "metrics": _json(check.get("metrics", {})),
                "thresholds": _json(check.get("thresholds", {})),
                "version": check.get("version", VERSION), "confidence": check.get("confidence", ""),
                "events": _json([event for event in result.get("events", [])
                                 if event.get("detector") == check.get("name")]),
            })
    return stream.getvalue()


def _text(value: Any) -> str:
    return escape(str(value), quote=True)


def _pre(value: Any) -> str:
    return "<pre>" + _text(_json(value, indent=2)) + "</pre>"


def _evidence(path: str) -> str:
    parsed = urlsplit(path)
    parts = PurePosixPath(path).parts
    if (not path or parsed.scheme or parsed.netloc or parsed.query or parsed.fragment
            or path.startswith("/") or "\\" in path or ".." in parts):
        return "<span>Invalid evidence path: " + _text(path) + "</span>"
    href = escape(quote(path, safe="/"), quote=True)
    return f'<a href="{href}"><img src="{href}" alt="{_text(path)}" loading="lazy"></a>'


def _html(results: list[dict], policy: dict[str, str], generated: str) -> str:
    sections = []
    for result in results:
        rows = []
        for check in _checks(result):
            rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in (
                _text(check.get("name", "")), _text(check.get("status", "ERROR")),
                _text(check.get("message", "")), _pre(check.get("metrics", {})),
                _pre(check.get("thresholds", {})), _text(check.get("version", VERSION)),
                _text(check.get("confidence", "")),
            )) + "</tr>")
        events = []
        for event in result.get("events", []):
            details = {key: value for key, value in event.items() if key != "evidence"}
            events.append('<article class="event"><h4>' + _text(event.get("detector", "event"))
                          + "</h4>" + _pre(details) + '<div class="evidence">'
                          + "".join(_evidence(str(path)) for path in event.get("evidence", []))
                          + "</div></article>")
        sections.append("<section><h2>" + _text(result.get("file_id", "unknown")) + " · "
                        + _text(result.get("status", "ERROR")) + "</h2><p>" + _text(result.get("path", ""))
                        + " · " + _text(result.get("codec", "")) + " · Frames: "
                        + _text(result.get("frame_count", 0)) + "</p><h3>Checks</h3>"
                        + '<div class="table"><table><thead><tr><th>Check</th><th>Status</th>'
                        + '<th>Message</th><th>Metrics</th><th>Thresholds</th><th>Version</th><th>Confidence</th>'
                        + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
                        + "<h3>Events</h3>" + ("".join(events) or "<p>No events.</p>")
                        + "<h3>Metadata / timestamp provenance</h3>" + _pre(result.get("metadata", {}))
                        + "<h3>Diagnostics</h3>" + _pre(result.get("diagnostics", []))
                        + "<h3>Performance</h3>" + _pre(result.get("performance", {})) + "</section>")
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>VideoQA V1 report</title><style>
body{font:15px system-ui,sans-serif;color:#203047;background:#f3f6fa;margin:0;padding:24px}
main{max-width:1500px;margin:auto}section{background:white;border:1px solid #d9e1ea;padding:22px;margin:20px 0;border-radius:10px}
h1,h2,h3{line-height:1.3}table{border-collapse:collapse;width:100%}td,th{border:1px solid #d9e1ea;text-align:left;padding:8px;vertical-align:top}
th{background:#edf2f7}.table{overflow:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;margin:0;max-width:650px}
.event{border-top:1px solid #d9e1ea;padding:12px 0}.evidence{display:flex;flex-wrap:wrap;gap:10px;margin-top:12px}
.evidence img{max-width:300px;max-height:200px;object-fit:contain}a{color:#235baa}
</style></head><body><main><h1>VideoQA V1 report</h1>""" + (
        "<p>Generated: " + _text(generated) + " · Analyzer: " + VERSION + "</p><p>CI policy: "
        + _text(_json(policy)) + ". SUSPECT and UNVERIFIABLE retain their diagnostic status.</p>"
        + "".join(sections) + "</main></body></html>\n"
    )


def write_reports(results: list[dict], output_dir: Path, policy: dict | None = None) -> None:
    """Write four formats; uncertainty fails CI by default and can explicitly warn."""
    resolved_policy = _policy(policy)
    for result in results:
        for check in _checks(result):
            if check.get("status", "ERROR") not in STATUSES:
                raise ValueError(f"Unknown check status: {check.get('status')}")
    generated = datetime.now(timezone.utc).isoformat()
    # Serialize all outputs first so non-finite metrics fail before files are written.
    documents = {
        "report.json": _json({"schema_version": "1", "analyzer_version": VERSION,
                              "generated_at": generated, "policy": resolved_policy,
                              "results": results}, indent=2) + "\n",
        "report.csv": _csv(results),
        "report.html": _html(results, resolved_policy, generated),
        "junit.xml": _junit(results, resolved_policy),
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in documents.items():
        temporary = output_dir / (filename + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(output_dir / filename)
