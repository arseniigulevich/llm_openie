"""Config loading and saving; loading of predictions, gold and MINE graphs; triple stats."""

import json
from pathlib import Path

import pytest

from eval_config import EvalConfig, MetricsConfig
from evaluation.inputs import load_gold, load_system_inputs, triple_stats
from tests.conftest import ROOT, TEXT_1, sanitize, write_config


def test_config_paths_defaults_and_round_trip(tmp_path, toy_dataset):
    cfg_path = write_config(tmp_path, toy_dataset, "http://x/v1/", {"FS": True, "recompute_FS": True},
                            extra_dataset='[datasets.toy.system_paths]\ngamma = "rel/gamma.json"')
    config = EvalConfig.from_toml(cfg_path, base_dir=tmp_path)
    ds = config.datasets["toy"]
    assert ds.get_extraction_path("alpha") == toy_dataset / "toy_alpha.json"
    assert ds.get_extraction_path("gamma") == tmp_path / "rel" / "gamma.json"
    assert config.metrics.GS is True                     # a metric missing from the TOML is on
    assert config.metrics.should_recompute("FS") and not config.metrics.should_recompute("S-r")

    saved = tmp_path / "saved.toml"
    config.save_config(saved)
    again = EvalConfig.from_toml(saved, base_dir=Path("/elsewhere"))
    assert again.to_dict() == config.to_dict()           # mine flag and system_paths survive


def test_dataset_without_gold(tmp_path):
    cfg = tmp_path / "c.toml"
    cfg.write_text('[datasets.mine]\nmine = true\nextractions_dir = "d"\n')
    ds = EvalConfig.from_toml(cfg, base_dir=tmp_path).datasets["mine"]
    assert ds.ground_truth is None and ds.mine and load_gold(ds) is None


def test_metric_aliases():
    m = MetricsConfig.from_dict({"SS_recompute": True})  # older spelling
    assert m.should_recompute("S-f1") and m.should_recompute("SS") and not m.should_recompute("CS-r")


def test_inputs_sanitise_text_keys_of_gold_and_predictions(tmp_path, toy_dataset):
    config = EvalConfig.from_toml(write_config(tmp_path, toy_dataset, "http://x/v1/", {}))
    ds = config.datasets["toy"]
    inputs = load_system_inputs("alpha", ds, load_gold(ds))
    assert TEXT_1 not in inputs.predictions and sanitize(TEXT_1) in inputs.predictions
    assert sanitize(TEXT_1) in inputs.gold_keyed and TEXT_1 in inputs.gold
    # the trailing confidence is stripped
    assert inputs.predictions[sanitize(TEXT_1)][0] == ["Marie Curie", "won", "the Nobel Prize"]
    assert load_system_inputs("missing", ds, None) is None


def test_triple_stats():
    dataset, per = triple_stats({"a": [["x y", "r", "z"], ["x", "r s", "z", 0.5]], "b": []})
    assert per["a"] == {"num_triples": 2, "avg_tokens_per_triple": 4.0}
    assert per["b"] == {"num_triples": 0, "avg_tokens_per_triple": 0}
    assert dataset == {"avg_num_triples": 1.0, "avg_num_tokens_per_triple": 2.0}


def test_mine_graphs_become_text_triples():
    from eval_config import DatasetConfig
    ds = DatasetConfig(name="mine", display_name="MINE", ground_truth=None,
                       extractions_dir=ROOT / "data" / "MINE", extraction_template="{system}_mine.json",
                       lda_model="mine", systems=["llm"], mine=True)
    inputs = load_system_inputs("llm", ds, None)
    graphs = json.loads((ROOT / "data" / "MINE" / "llm_mine.json").read_text())
    first = inputs.mine_dataset["1"]["essay_content"]
    assert inputs.predictions[sanitize(first)][0] == graphs["1"]["relations"][0][:3]
    assert len(inputs.predictions) == len([i for i in graphs if i in inputs.mine_dataset])
