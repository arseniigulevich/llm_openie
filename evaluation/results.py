"""Per-system result files ``results/<dataset>/<system>.json``: loading, incremental and
atomic saving, and the compact form that drops bulky per-pair details."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np

# Per-sentence entailment keys kept in compact results.
ENTAILMENT_SENTENCE_KEYS = (
    "sentence_r", "sentence_p", "sentence_f1",
    "combined_tuple_r", "combined_tuple_p", "combined_tuple_f1",
    "tuple_r", "tuple_p", "tuple_f1",
)


def to_serializable(obj: Any) -> Any:
    """Convert numpy scalars, tuples and sets into JSON-serialisable values."""
    if isinstance(obj, dict):
        return {key: to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_serializable(value) for value in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def compact(results: Dict[str, Any]) -> Dict[str, Any]:
    """Drop every ``*_details`` key and keep only the aggregate per-sentence entailment
    keys. ``results`` must already be serialisable."""
    out: Dict[str, Any] = {}
    for dataset, ds in results.items():
        if not isinstance(ds, dict):
            out[dataset] = ds
            continue
        ds_out: Dict[str, Any] = {}
        for key, value in ds.items():
            if key.endswith("_details"):
                continue
            if key == "per_sentence" and isinstance(value, dict):
                value = {
                    sent: ({**m, "Entailment": {k: m["Entailment"][k] for k in ENTAILMENT_SENTENCE_KEYS
                                                 if k in m["Entailment"]}}
                           if isinstance(m, dict) and isinstance(m.get("Entailment"), dict) else m)
                    for sent, m in value.items()
                }
            ds_out[key] = value
        out[dataset] = ds_out
    return out


def write_json_atomic(path: Path, payload: Any, indent: Optional[int] = 4) -> None:
    """Write JSON via a temporary sibling file so an interrupted write cannot leave a
    truncated results file (which would break resuming)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=indent)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


class ResultStore:
    """The results of one system on one dataset, kept in memory and saved after every
    metric. ``dataset`` holds the dataset-level scores, ``per_sentence`` the per-text
    entries keyed by the sanitised source text."""

    def __init__(self, path: Path, dataset_name: str, compact_results: bool = False):
        self.path = path
        self.dataset_name = dataset_name
        self.compact_results = compact_results
        self.data: Dict[str, Any] = {}
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        self.dataset: Dict[str, Any] = self.data.setdefault(dataset_name, {})
        self.per_sentence: Dict[str, Dict[str, Any]] = self.dataset.setdefault("per_sentence", {})

    def has(self, keys: Iterable[str], sentences: Iterable[str] = (), per_sentence: bool = True) -> bool:
        """True when every key is present at dataset level and, if ``per_sentence``, in
        the entry of every sentence."""
        keys = list(keys)
        if any(k not in self.dataset for k in keys):
            return False
        if not per_sentence:
            return True
        return all(k in self.per_sentence.get(s, {}) for s in sentences for k in keys)

    def update(self, dataset_values: Dict[str, Any],
               per_sentence: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        self.dataset.update(dataset_values)
        for sent, values in (per_sentence or {}).items():
            self.per_sentence.setdefault(sent, {}).update(values)

    def payload(self, compact_results: Optional[bool] = None) -> Dict[str, Any]:
        payload = to_serializable(self.data)
        use_compact = self.compact_results if compact_results is None else compact_results
        return compact(payload) if use_compact else payload

    def save(self) -> None:
        write_json_atomic(self.path, self.payload())
