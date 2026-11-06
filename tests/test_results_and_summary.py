"""Result files, compact form, and the run summary."""

import json

import numpy as np

from evaluation import summary as status
from evaluation.results import ResultStore, compact, to_serializable
from evaluation.summary import RunSummary


def test_to_serializable_converts_numpy_and_containers():
    out = to_serializable({"a": np.float32(0.5), "b": (1, np.int64(2)), "c": {np.bool_(True)}})
    assert out == {"a": 0.5, "b": [1, 2], "c": [True]}
    json.dumps(out)


def test_compact_drops_details_and_trims_entailment():
    results = {"ds": {
        "FS": 0.5, "FS_details": {"x": 1}, "Entailment_sentence_details": {},
        "per_sentence": {
            "s1": {"FS": 0.5, "Entailment": {"sentence_p": 1.0, "pairs": [1, 2], "tuple_f1": 0.2}},
            "s2": {"Entailment": None},
        },
    }}
    out = compact(results)["ds"]
    assert set(out) == {"FS", "per_sentence"}
    assert out["per_sentence"]["s1"]["Entailment"] == {"sentence_p": 1.0, "tuple_f1": 0.2}
    assert out["per_sentence"]["s2"] == {"Entailment": None}


def test_store_round_trip_and_has(tmp_path):
    path = tmp_path / "sys.json"
    store = ResultStore(path, "ds", compact_results=True)
    store.update({"FS": 0.5, "FS_details": {"a": 1}}, {"s1": {"FS": 0.5}, "s2": {"FS": None}})
    store.save()
    assert not list(tmp_path.glob("*.tmp.*"))  # the temporary file was renamed into place
    reloaded = ResultStore(path, "ds", compact_results=True)
    assert "FS_details" not in reloaded.dataset
    assert reloaded.has(["FS"], ["s1", "s2"])
    assert not reloaded.has(["FS"], ["s1", "s3"])          # a sentence without the key
    assert not reloaded.has(["GS"], [])
    assert reloaded.has(["FS"], ["s3"], per_sentence=False)


def test_store_keeps_other_datasets(tmp_path):
    path = tmp_path / "sys.json"
    path.write_text(json.dumps({"other": {"FS": 1.0, "per_sentence": {}}}))
    store = ResultStore(path, "ds")
    store.update({"GS": 0.3})
    store.save()
    data = json.loads(path.read_text())
    assert data["other"]["FS"] == 1.0 and data["ds"]["GS"] == 0.3


def test_summary_status_priority_and_payload(tmp_path):
    s = RunSummary("cfg.toml")
    s.mark("ds", "sys", "FS", status.FAILED, "boom")
    s.mark("ds", "sys", "FS", status.FINISHED_PREVIOUS)    # lower priority: ignored
    s.mark("ds", "sys", "GS", status.FINISHED_PREVIOUS)
    s.mark("ds", "sys", "GS", status.FINISHED_THIS)
    s.mark("ds", "sys", "CS-r", status.SKIPPED, "No ground truth")
    s.add_system_error("ds", "sys2", "file missing")
    s.write(tmp_path / "summary.json")
    payload = json.loads((tmp_path / "summary.json").read_text())
    sys_entry = payload["datasets"]["ds"]["systems"]["sys"]
    assert sys_entry["failed"] == [{"metric": "FS", "error": "boom"}]
    assert sys_entry["finished_this_run"] == ["GS"]
    assert sys_entry["skipped"] == ["CS-r"]
    assert payload["datasets"]["ds"]["systems"]["sys2"]["system_errors"] == ["file missing"]
