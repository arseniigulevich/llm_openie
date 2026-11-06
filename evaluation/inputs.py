"""Inputs of a system on a dataset: predictions, gold triples, MINE graphs, the triple
statistics, and the embedding cache that US and CS read."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from nltk.tokenize import word_tokenize

from eval_config import DatasetConfig, EvalConfig
from source.converter import _sanitize_field, normalize_prediction_triples
from source.embeddings import (
    fetch_missing_embeddings,
    load_element_embeddings,
    prefetch_dataset_embeddings,
    save_embeddings_atomic,
)


@dataclass
class SystemInputs:
    """Everything the metrics need for one system on one dataset."""
    system: str
    predictions: Dict[str, List[List[str]]]       # {sanitised text: [[s, r, o, ...], ...]}
    gold: Optional[Dict[str, Any]] = None          # gold as stored in the file
    gold_keyed: Optional[Dict[str, Any]] = None    # gold keyed by the sanitised text
    mine_dataset: Optional[Dict[str, Dict[str, Any]]] = None
    mine_extractions: Optional[Dict[str, Dict[str, Any]]] = None
    sentences: List[str] = field(default_factory=list)


def load_gold(dataset_config: DatasetConfig) -> Optional[Dict[str, Any]]:
    path = dataset_config.ground_truth
    if path is None:
        return None
    if not path.exists():
        print(f"Warning: ground-truth file not found: {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_system_inputs(system: str, dataset_config: DatasetConfig,
                       gold: Optional[Dict[str, Any]]) -> Optional[SystemInputs]:
    """Load and normalise a system's extractions; None if the file is missing.

    MINE extractions are graph-form ({id: {entities, edges, relations}}); they are
    converted to {essay text: triples} so the generic metrics run unchanged, and the raw
    graphs are kept for the MINE score.
    """
    path = dataset_config.get_extraction_path(system)
    if not path.exists():
        return None
    mine_dataset = mine_extractions = None
    if dataset_config.mine:
        from source import mine_eval
        mine_dataset = mine_eval.load_mine_dataset(dataset_config.extractions_dir)
        mine_extractions = mine_eval.load_mine_extractions(path)
        raw = mine_eval.mine_extractions_to_text_triples(mine_extractions, mine_dataset)
    else:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    predictions = normalize_prediction_triples(raw)
    # CS matches gold to predictions by text, so the gold must be keyed by the same
    # sanitisation that normalize_prediction_triples applies to the prediction keys
    # (it changes, e.g., the newlines in the WiRe57 documents).
    gold_keyed = {_sanitize_field(k): v for k, v in gold.items()} if gold is not None else None
    return SystemInputs(
        system=system, predictions=predictions, gold=gold, gold_keyed=gold_keyed,
        mine_dataset=mine_dataset, mine_extractions=mine_extractions,
        sentences=list(predictions.keys()),
    )


def triple_stats(predictions: Dict[str, List[List[Any]]]):
    """Number of triples and mean word tokens per triple, per text and on average.

    Returns ``(dataset_values, per_sentence)``. A trailing numeric element (a confidence)
    is not counted; a text without triples has 0 tokens per triple.
    """
    per_sentence: Dict[str, Dict[str, Any]] = {}
    counts, token_means = [], []
    for text, triples in predictions.items():
        tokens_per_triple = []
        for triple in triples:
            elements = triple
            if triple and isinstance(triple[-1], (int, float, np.number)):
                elements = triple[:-1]
            tokens_per_triple.append(sum(len(word_tokenize(e)) for e in elements if isinstance(e, str)))
        mean_tokens = sum(tokens_per_triple) / len(tokens_per_triple) if tokens_per_triple else 0
        counts.append(len(triples))
        token_means.append(mean_tokens)
        per_sentence[text] = {"num_triples": len(triples), "avg_tokens_per_triple": mean_tokens}
    dataset_values = {
        "avg_num_triples": float(np.mean(counts)) if counts else 0.0,
        "avg_num_tokens_per_triple": float(np.mean(token_means)) if token_means else 0.0,
    }
    return dataset_values, per_sentence


def _triple_element_strings(predictions: Dict[str, List[List[Any]]]) -> set:
    return {el.strip() for triples in predictions.values() for t in triples
            for el in t[:3] if isinstance(el, str) and el.strip()}


def prepare_embeddings(dataset_config: DatasetConfig, config: EvalConfig,
                       systems: Iterable[str], embeddings_path: Optional[Path]) -> None:
    """Fetch the element embeddings missing from the cache (blocking; run in a thread).

    Only runs when an embedding endpoint is configured and US or CS is requested.
    """
    if not (config.emb_api_endpoint and (config.metrics.CS or config.metrics.US)):
        return
    if dataset_config.mine:
        # Graph-form MINE files are not {text: triples}; collect the element strings from
        # the converted predictions, which are the keys US looks up.
        strings: set = set()
        for system in systems:
            inputs = load_system_inputs(system, dataset_config, None)
            if inputs is not None:
                strings |= _triple_element_strings(inputs.predictions)
        emb_dict = load_element_embeddings()
        added = fetch_missing_embeddings(strings, emb_dict, endpoint=config.emb_api_endpoint,
                                         model=config.emb_api_model)
        print(f"  MINE: fetched {added} new embeddings for {len(strings)} unique triple elements")
        if added > 0 and config.emb_api_save_new and embeddings_path:
            try:
                save_embeddings_atomic(embeddings_path, emb_dict)
                print(f"  Saved updated embeddings to {embeddings_path}")
            except Exception as exc:
                print(f"  Warning: could not save embeddings to {embeddings_path}: {exc}")
        return
    prefetch_dataset_embeddings(
        [dataset_config.get_extraction_path(s) for s in systems],
        load_gold(dataset_config),
        load_element_embeddings(),
        endpoint=config.emb_api_endpoint,
        model=config.emb_api_model,
        save_path=embeddings_path if config.emb_api_save_new else None,
    )
