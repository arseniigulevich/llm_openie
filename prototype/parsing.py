"""Parsers that turn raw LLM stage output into Python structures."""

from __future__ import annotations

import json
import re

Triple = list  # [subject, predicate, object]

_NUMBERING = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


def _clean_line(line: str) -> str:
    return _NUMBERING.sub("", line.strip()).strip()


def _extract_json_list(s: str) -> list | None:
    """Return the first ``[...]`` JSON list found in ``s``, or ``None``."""
    i, j = s.find("["), s.rfind("]")
    if i == -1 or j == -1 or j < i:
        return None
    try:
        value = json.loads(s[i : j + 1])
    except Exception:
        return None
    return value if isinstance(value, list) else None


def parse_lines(text: str, *, drop_labels: tuple[str, ...] = ()) -> list[str]:
    out, seen = [], set()
    lowered_drop = {d.lower() for d in drop_labels}
    for raw in text.splitlines():
        line = _clean_line(raw)
        if not line or line.lower() in lowered_drop:
            continue
        if line not in seen:
            seen.add(line)
            out.append(line)
    return out


def parse_atomic(text: str) -> list[str]:
    # Keep order and duplicates collapsed; atomic sentences are kept verbatim.
    return parse_lines(text, drop_labels=("atomic sentences:",))


def parse_entities(text: str) -> list[str]:
    return parse_lines(text, drop_labels=("entities:",))


def parse_triples(text: str) -> list[Triple]:
    """Parse one JSON triple per line; tolerate a legacy ``(s | p | o)`` form."""
    out: list[Triple] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        lst = _extract_json_list(line)
        if lst and len(lst) >= 3:
            s, p, o = (str(lst[0]).strip(), str(lst[1]).strip(), str(lst[2]).strip())
            if s and p and o:
                out.append([s, p, o])
        elif "|" in line:
            parts = [x.strip(" ()") for x in line.split("|")]
            if len(parts) >= 3 and all(parts[:3]):
                out.append([parts[0], parts[1], parts[2]])
    return _dedupe(out)


def parse_verify(text: str) -> tuple[list[Triple], list[Triple]]:
    """Return (supported triples, missing triples) from a verification response."""
    supported: list[Triple] = []
    missing: list[Triple] = []
    section: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith("support"):
            section = "support"
            continue
        if low.startswith("missing"):
            section = "missing"
            continue
        lst = _extract_json_list(line)
        if not lst or len(lst) < 3:
            continue
        triple = [str(lst[0]).strip(), str(lst[1]).strip(), str(lst[2]).strip()]
        if not all(triple):
            continue
        if section == "support":
            after = line[line.rfind("]") + 1 :].lower()
            if "supported" in after and "not supported" not in after:
                supported.append(triple)
        elif section == "missing":
            missing.append(triple)
    return _dedupe(supported), _dedupe(missing)


def _dedupe(triples: list[Triple]) -> list[Triple]:
    out, seen = [], set()
    for t in triples:
        key = tuple(t)
        if key not in seen:
            seen.add(key)
            out.append(list(t))
    return out
