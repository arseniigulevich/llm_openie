"""End-to-end runs of ``evaluate.py`` on the toy dataset against the stub endpoint."""

import json
import math
import sys

import pytest

from evaluation.cli import main
from tests import llm_stub
from tests.conftest import ALL_OFF, ROOT, TEXT_1, TEXT_2, TEXT_3, sanitize, write_config

FAST = {**ALL_OFF, "TS": True, "US": True, "CS": True, "FS": True, "GS": True, "Coherence": True}
T1, T2, T3 = sanitize(TEXT_1), sanitize(TEXT_2), sanitize(TEXT_3)


def _results(tmp_path, system):
    return json.loads((tmp_path / "results" / "toy" / f"{system}.json").read_text())["toy"]


def _summary(tmp_path):
    return json.loads((tmp_path / "results" / "evaluation_run_summary.json").read_text())["datasets"]["toy"]["systems"]


def _fs(triples):
    labels = [llm_stub.fs_label(json.dumps(t)) for t in triples]
    return sum(labels) / len(labels)


@pytest.fixture
def run(tmp_path, toy_dataset, llm_server):
    def _run(metrics=FAST, *args, **kwargs):
        cfg = write_config(tmp_path, toy_dataset, llm_server.url, metrics, **kwargs)
        assert main(["--config", str(cfg), *args]) == 0
        return cfg
    return _run


def test_full_run_scores(tmp_path, run):
    run()
    alpha, beta = _results(tmp_path, "alpha"), _results(tmp_path, "beta")
    a1 = [["Marie Curie", "won", "the Nobel Prize"]] * 2 + [["Marie Curie", "was born in", "Warsaw"]]
    a2 = [["The river", "flows into", "the sea"]]

    # stats
    assert alpha["per_sentence"][T1]["num_triples"] == 3 and alpha["per_sentence"][T3]["num_triples"] == 0
    assert alpha["avg_num_triples"] == pytest.approx(4 / 3)
    # uniqueness: one repeated triple among three -> 4 of 6 ordered pairs differ
    assert alpha["per_sentence"][T1]["US"] == pytest.approx(4 / 6)
    assert alpha["per_sentence"][T2]["US"] == 1.0 and alpha["per_sentence"][T3]["US"] == 0
    # completeness: the gold of TEXT_1 is keyed with a newline and still matches
    assert alpha["per_sentence"][T1]["CS-r"] == 1.0 and alpha["per_sentence"][T1]["CS-p"] == 1.0
    assert alpha["per_sentence"][T3]["CS-r"] is None           # no gold units: excluded
    assert beta["per_sentence"][T1]["CS-r"] == 0.0
    assert beta["per_sentence"][T2]["CS-r"] == 1.0 and beta["per_sentence"][T2]["CS-p"] == 0.5
    assert alpha["CS-r"] == 1.0 and beta["CS-p"] == pytest.approx(0.25)
    # LLM judges, against the stub judge's labels
    assert alpha["per_sentence"][T1]["FS"] == pytest.approx(_fs(a1))
    assert alpha["per_sentence"][T3]["FS"] is None             # empty set: no FS
    assert alpha["FS"] == pytest.approx((_fs(a1) + _fs(a2)) / 2)
    gs1 = sum(math.exp(-llm_stub.gs_split(json.dumps(t))) for t in a1[1:]) / 2  # duplicates once
    assert alpha["per_sentence"][T1]["GS"] == pytest.approx(gs1)
    assert alpha["per_sentence"][T3]["Coherence"] == 0.0       # empty set counts as 0
    # TS with the WiRe57 LDA model
    assert 0 < alpha["TS"] <= 1 and set(alpha["per_sentence"]) == {T1, T2, T3}
    # compact results: no details
    assert not any(k.endswith("_details") for k in alpha)

    summary = _summary(tmp_path)
    for system in ("alpha", "beta"):
        assert summary[system]["failed"] == [] and summary[system]["system_errors"] == []
        assert set(summary[system]["finished_this_run"]) >= {"TS", "US", "CS-r", "CS-p", "FS", "GS", "Coherence"}
    saved = (tmp_path / "results" / "toy" / "eval_config.toml").read_text()
    assert "toy_{system}.json" in saved


def test_rerun_resumes_without_requests(tmp_path, run, llm_server):
    run()
    before = llm_server.requests
    first = _results(tmp_path, "alpha")
    run()
    assert llm_server.requests == before
    assert _results(tmp_path, "alpha") == first
    summary = _summary(tmp_path)
    assert summary["alpha"]["finished_this_run"] == []
    assert "FS" in summary["alpha"]["finished_previous_run"]


def test_recompute_flag_and_partial_results(tmp_path, run, llm_server):
    run()
    # a sentence that lost its FS entry makes FS run again (for this system only)
    path = tmp_path / "results" / "toy" / "beta.json"
    data = json.loads(path.read_text())
    del data["toy"]["per_sentence"][T2]["FS"]
    path.write_text(json.dumps(data))
    before = llm_server.requests
    run()
    assert llm_server.requests - before == 4                      # beta's four triples
    assert _summary(tmp_path)["beta"]["finished_this_run"] == ["FS"]

    before = llm_server.requests
    run({**FAST, "recompute_GS": True})
    assert llm_server.requests - before == 4 + 4                  # every triple is judged, duplicates too
    assert all(_summary(tmp_path)[s]["finished_this_run"] == ["GS"] for s in ("alpha", "beta"))


def test_failures_are_isolated(tmp_path, run, toy_dataset):
    # a missing LDA model fails TS only; a missing extraction file fails one system only
    run(FAST, "--systems", "alpha", "beta", "ghost",
        extra_dataset="", extra_general="")
    summary = _summary(tmp_path)
    assert summary["ghost"]["system_errors"] == ["extraction file not found"]
    assert summary["alpha"]["failed"] == []

    cfg = write_config(tmp_path, toy_dataset, "http://127.0.0.1:9/v1/", {**ALL_OFF, "TS": True, "US": True})
    text = cfg.read_text().replace('lda_model = "wire57"', 'lda_model = "no_such_model"')
    cfg.write_text(text.replace("TS = true", "TS = true\nrecompute_TS = true"))
    assert main(["--config", str(cfg)]) == 0
    summary = _summary(tmp_path)
    assert summary["alpha"]["failed"][0]["metric"] == "TS"
    assert "Missing LDA" in summary["alpha"]["failed"][0]["error"]
    assert "US" in summary["alpha"]["finished_previous_run"]


def test_cs_without_gold_is_skipped(tmp_path, toy_dataset, llm_server):
    cfg = write_config(tmp_path, toy_dataset, llm_server.url, {**ALL_OFF, "CS": True})
    cfg.write_text("\n".join(l for l in cfg.read_text().splitlines() if not l.startswith("ground_truth")))
    assert main(["--config", str(cfg)]) == 0
    assert _summary(tmp_path)["alpha"]["skipped"] == ["CS-p", "CS-r"]
    assert "CS-r" not in _results(tmp_path, "alpha")


def test_descriptive_subset_on_run_output(tmp_path, run, monkeypatch):
    # US, FS, Coherence and the surprisal F1 enter the subset score; S-f1 is written by
    # hand here so the test does not need the surprisal model
    cfg = run()
    for system in ("alpha", "beta"):
        path = tmp_path / "results" / "toy" / f"{system}.json"
        data = json.loads(path.read_text())
        for entry in data["toy"]["per_sentence"].values():
            entry["S-f1"] = 0.5
        path.write_text(json.dumps(data))
    sys.path.insert(0, str(ROOT))
    import descriptive_subset
    monkeypatch.setattr(sys, "argv", ["descriptive_subset.py", "--config", str(cfg)])
    descriptive_subset.main()
    report = json.loads((tmp_path / "results" / "descriptive_subset.json").read_text())["toy"]
    alpha = _results(tmp_path, "alpha")["per_sentence"]
    expected = [(e["US"] + e["FS"] + e["Coherence"] + 0.5) / 4 for e in alpha.values() if e["num_triples"]]
    assert report["alpha"]["Subset"] == pytest.approx(sum(expected) / len(expected))
    cs = [2 * e["CS-p"] * e["CS-r"] / (e["CS-p"] + e["CS-r"]) if e["CS-p"] + e["CS-r"] else 0.0
          for e in alpha.values() if e["CS-r"] is not None]
    assert report["alpha"]["CS-f1"] == pytest.approx(sum(cs) / len(cs))


def test_missing_config_returns_error(tmp_path):
    assert main(["--config", str(tmp_path / "nope.toml")]) == 2


@pytest.mark.slow
def test_mine_end_to_end(tmp_path, llm_server):
    """MINE with KGGen deduplication and the stub judge, on three articles; a limited run
    is not taken as complete by a later run with more articles."""
    cfg = tmp_path / "mine.toml"
    cfg.write_text(f"""
[general]
results_dir = "{tmp_path / 'results'}"
llm_endpoint = "{llm_server.url}"
max_concurrent_systems = 1
save_stats_separate = false
compact_results = true
[embeddings]
path = "{tmp_path / 'unused.json'}"
[metrics]
TS = false
US = false
CS = false
FS = false
GS = false
Coherence = false
SS = false
Entailment = false
MINE = true
[datasets.mine]
mine = true
extractions_dir = "{ROOT / 'data' / 'MINE'}"
extraction_template = "{{system}}_mine.json"
systems = ["llm"]
""")
    assert main(["--config", str(cfg), "--mine-max-articles", "2"]) == 0
    res = json.loads((tmp_path / "results" / "mine" / "llm.json").read_text())["mine"]
    assert res["MINE_num_articles"] == 2
    scored = [v["MINE"] for v in res["per_sentence"].values() if "MINE" in v]
    assert len(scored) == 2 and res["MINE"] == pytest.approx(sum(scored) / 2)
    assert all(0 <= s <= 1 for s in scored)

    assert main(["--config", str(cfg), "--mine-max-articles", "3"]) == 0
    res = json.loads((tmp_path / "results" / "mine" / "llm.json").read_text())["mine"]
    assert res["MINE_num_articles"] == 3


@pytest.mark.slow
def test_entailment_and_surprisal_end_to_end(tmp_path, run):
    """The model-based metrics: the cross-encoder is unloaded before surprisal runs, and a
    rerun with compact results does not recompute Entailment."""
    from source import entailment_eval
    metrics = {**ALL_OFF, "Entailment": True, "SS": True}
    run(metrics, extra_general='ss_model = "gpt2"\nentailment_batch_size = 16')
    assert entailment_eval.sota_model is None
    alpha = _results(tmp_path, "alpha")
    assert set(alpha["Entailment"]) == {"sentence", "combined_tuple", "tuple"}
    assert all(0 <= alpha["per_sentence"][t]["S-f1"] <= 1 for t in (T1, T2))
    run(metrics, extra_general='ss_model = "gpt2"\nentailment_batch_size = 16')
    assert _summary(tmp_path)["alpha"]["finished_this_run"] == []
