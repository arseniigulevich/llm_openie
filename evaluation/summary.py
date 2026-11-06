"""Run summary: the status of every metric per dataset and system
(``evaluation_run_summary.json``)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

NOT_REQUESTED = "not_requested"
SKIPPED = "skipped"                      # not applicable, e.g. CS without gold triples
FINISHED_PREVIOUS = "finished_previous_run"
FINISHED_THIS = "finished_this_run"
FAILED = "failed"

# A later status overwrites an earlier one only if it has at least the same priority,
# so a failure in one phase is not hidden by a skip in the next.
_PRIORITY = {NOT_REQUESTED: 0, SKIPPED: 1, FINISHED_PREVIOUS: 2, FINISHED_THIS: 3, FAILED: 4}


def short_error(exc: BaseException, max_length: int = 500) -> str:
    msg = f"{type(exc).__name__}: {exc}"
    return msg if len(msg) <= max_length else msg[:max_length - 3] + "..."


class RunSummary:
    def __init__(self, config_path: str = ""):
        self.started_at_utc = datetime.now(timezone.utc).isoformat()
        self.config_path = config_path
        self.llm_model_info: Optional[Dict[str, Any]] = None
        self.datasets: Dict[str, Dict[str, Any]] = {}

    def _system(self, dataset: str, system: str) -> Dict[str, Any]:
        systems = self.datasets.setdefault(dataset, {}).setdefault("systems", {})
        return systems.setdefault(system, {"metrics": {}, "errors": []})

    def mark(self, dataset: str, system: str, metric: str, status: str,
             error: Optional[str] = None) -> None:
        metrics = self._system(dataset, system)["metrics"]
        current = metrics.get(metric, {}).get("status")
        if current is not None and _PRIORITY.get(status, -1) < _PRIORITY.get(current, -1):
            return
        entry = {"status": status}
        if error:
            entry["error"] = error
        metrics[metric] = entry

    def status(self, dataset: str, system: str, metric: str) -> Optional[str]:
        return self._system(dataset, system)["metrics"].get(metric, {}).get("status")

    def add_system_error(self, dataset: str, system: str, error: str) -> None:
        self._system(dataset, system)["errors"].append(error)
        self.mark(dataset, system, "__system__", FAILED, error)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "started_at_utc": self.started_at_utc,
            "config_path": self.config_path,
            "datasets": {},
        }
        if self.llm_model_info is not None:
            payload["llm_model_info"] = self.llm_model_info
        for dataset, entry in self.datasets.items():
            out = {"systems": {}}
            for system, sys_entry in entry.get("systems", {}).items():
                metrics = sys_entry["metrics"]
                by_status = lambda s: sorted(m for m, i in metrics.items() if i["status"] == s)
                out["systems"][system] = {
                    "finished_previous_run": by_status(FINISHED_PREVIOUS),
                    "finished_this_run": by_status(FINISHED_THIS),
                    "skipped": by_status(SKIPPED),
                    "failed": [{"metric": m, "error": i.get("error", "")}
                               for m, i in metrics.items() if i["status"] == FAILED],
                    "system_errors": list(sys_entry["errors"]),
                    "metrics": metrics,
                }
            payload["datasets"][dataset] = out
        return payload

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
