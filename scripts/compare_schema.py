"""Evidence-aware model comparison validation, separate from research cards."""
from datetime import date
from math import isfinite
from pathlib import PurePosixPath
from urllib.parse import urlsplit


def validate_compare(data):
    errors = []

    def require(ok, message):
        if not ok:
            errors.append(message)

    def iso(value):
        try:
            return isinstance(value, str) and date.fromisoformat(value).isoformat() == value
        except ValueError:
            return False

    def https(value):
        if not isinstance(value, str):
            return False
        try:
            parsed = urlsplit(value)
            return parsed.scheme == "https" and bool(parsed.hostname) and not parsed.username
        except ValueError:
            return False

    def strings(obj, keys, tag):
        for key in keys:
            require(isinstance(obj.get(key), str) and bool(obj[key].strip()), f"{tag}: missing {key}")

    require(data.get("schemaVersion") == 2, "schemaVersion must be 2")
    require(iso(data.get("updated")), "updated must be an ISO date")
    strings(data, ("note", "default"), "compare")
    metrics = data.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        return errors + ["metrics must be a nonempty list"]
    ids, evidence_types = set(), set()
    for m in metrics:
        if not isinstance(m, dict):
            errors.append("metric must be an object")
            continue
        tag = str(m.get("id", "?"))
        strings(m, ("id", "label", "benchmark", "version", "evaluator", "harness", "protocol", "budget", "note"), tag)
        require(tag not in ids, f"{tag}: duplicate metric id")
        ids.add(tag)
        kind = m.get("evidence")
        require(kind in ("independent", "vendor"), f"{tag}: invalid evidence")
        evidence_types.add(kind)
        require(m.get("unit") in ("index", "%"), f"{tag}: invalid unit")
        require(isinstance(m.get("comparable"), bool), f"{tag}: comparable must be boolean")
        require(not m.get("comparable") or (kind == "independent" and m.get("harness") != "mixed"), f"{tag}: vendor/mixed configurations must not rank")
        for key in ("sourceUrl", "methodologyUrl"):
            require(https(m.get(key)), f"{tag}: {key} must be HTTPS")
        captured = m.get("capturedAt")
        require(iso(captured), f"{tag}: capturedAt must be ISO date")
        if iso(captured) and iso(data.get("updated")):
            require(captured <= data["updated"], f"{tag}: capture after updated")
        for key in ("publishedAt", "evaluationDate"):
            require(key in m and (m[key] is None or iso(m[key])), f"{tag}: {key} must be date or explicit null")
            if iso(m.get(key)) and iso(captured):
                require(m[key] <= captured, f"{tag}: {key} after capture")
        items = m.get("items")
        if not isinstance(items, list) or not items:
            errors.append(f"{tag}: items must be nonempty")
            continue
        seen = set()
        for row in items:
            if not isinstance(row, dict):
                errors.append(f"{tag}: row must be an object")
                continue
            strings(row, ("model", "setting"), tag)
            require(set(row) <= {"model", "setting", "score", "sourceUrl", "harness", "missingReason"}, f"{tag}: unexpected row fields; split differing benchmark/evidence into a new group")
            require(https(row.get("sourceUrl")), f"{tag}: row sourceUrl must be HTTPS")
            key = (str(row.get("model")), str(row.get("setting")))
            require(key not in seen, f"{tag}: duplicate model/configuration")
            seen.add(key)
            if m.get("harness") == "mixed":
                strings(row, ("harness",), tag)
            elif "harness" in row:
                require(row["harness"] == m.get("harness"), f"{tag}: mismatched harness")
            require("score" in row, f"{tag}: missing score")
            score = row.get("score")
            if score is None:
                strings(row, ("missingReason",), tag)
            else:
                require(type(score) in (int, float) and isfinite(score) and 0 <= score <= 100, f"{tag}: score must be finite within 0..100 or null")
    require(evidence_types == {"independent", "vendor"}, "both evidence views need records")
    require(any(m.get("id") == data.get("default") and m.get("evidence") == "independent" for m in metrics if isinstance(m, dict)), "default must be an independent metric")
    archive = data.get("archive", {})
    require(isinstance(archive, dict), "archive must be object")
    if isinstance(archive, dict):
        strings(archive, ("url", "doc", "note"), "archive")
        require(iso(archive.get("date")), "archive date must be ISO")
        for key, prefix in (("url", "data/archive/"), ("doc", "docs/")):
            value = archive.get(key, "")
            require(isinstance(value, str) and value.startswith(prefix) and ".." not in PurePosixPath(value).parts, f"archive: invalid {key}")
    pending = data.get("pending")
    require(isinstance(pending, list), "pending must be list")
    for row in pending if isinstance(pending, list) else []:
        if not isinstance(row, dict):
            errors.append("pending row must be object")
            continue
        strings(row, ("model", "reason"), "pending")
        require(https(row.get("sourceUrl")), "pending sourceUrl must be HTTPS")
        require("score" not in row, "pending models must not have placeholder scores")
    return errors
