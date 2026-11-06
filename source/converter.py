"""Utilities for converting OpenIE JSON outputs into TSV files for entailment evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

Triple = List[str]
TripleDict = Dict[str, List[Triple]]


def _sanitize_field(value: str) -> str:
    """Remove tabs/newlines and strip extra whitespace."""
    return " ".join(str(value).replace("\t", " ").replace("\n", " ").split())


def _strip_confidence(elements: Sequence) -> list:
    """Remove a trailing confidence score from an extraction tuple.

    Classical systems append their confidence as a number: ["people", "do", 0.85].
    Only an int or float in the last slot is a confidence. A numeric *string* is an
    argument and is kept: ["X", "was formed in", "1886"] stays a three-slot triple.
    Every extraction file stores confidences as numbers (checked 2026-09-29).
    """
    elems = list(elements)
    last = elems[-1] if elems else None
    if len(elems) >= 2 and isinstance(last, (int, float)) and not isinstance(last, bool):
        return elems[:-1]
    return elems


def _normalize_triple(candidate: Sequence) -> Triple | None:
    """Convert an extraction tuple into a list of clean strings.

    Accepts tuples with 2 or more string arguments (subject, predicate,
    and optional objects).  A trailing numeric confidence score is stripped
    automatically.
    """
    if not isinstance(candidate, (list, tuple)) or len(candidate) < 2:
        return None
    stripped = _strip_confidence(candidate)
    if len(stripped) < 2:
        return None
    return [_sanitize_field(e) for e in stripped]


def _flatten_gold_entry(entry: Iterable) -> List[Triple]:
    """Return triples from gold entry.
    
    Handles two formats:
    - BenchIE format: List of clusters, each cluster is a list of synonym triples.
      We take the first triple from each cluster.
      Example: [[['subj1', 'pred1', 'obj1'], ['subj1_syn', 'pred1', 'obj1']], [['subj2', 'pred2', 'obj2']]]
    - LSOIE format: Simple list of triples (no clustering).
      Example: [['subj1', 'pred1', 'obj1'], ['subj2', 'pred2', 'obj2']]
    """
    flattened: List[Triple] = []
    if not isinstance(entry, Iterable) or isinstance(entry, (str, bytes)):
        return flattened
    
    entry_list = list(entry)
    if not entry_list:
        return flattened
    
    # Detect format by examining the first element
    first_item = entry_list[0]
    
    # Check if first_item is a triple (LSOIE format) or a cluster (BenchIE format)
    # A triple is a list/tuple of 2+ strings
    # A cluster is a list of triples (list of lists)
    is_flat_triple = (
        isinstance(first_item, (list, tuple)) and
        len(first_item) >= 2 and
        isinstance(first_item[0], str)
    )
    
    if is_flat_triple:
        # LSOIE format: entry is a list of triples directly
        for triple in entry_list:
            normalized = _normalize_triple(triple)
            if normalized:
                flattened.append(normalized)
    else:
        # BenchIE format: entry is a list of clusters
        for cluster in entry_list:
            if isinstance(cluster, Iterable) and not isinstance(cluster, (str, bytes)):
                cluster_list = list(cluster)
                if not cluster_list:
                    continue
                first = cluster_list[0]
                normalized = _normalize_triple(first)
                if normalized:
                    flattened.append(normalized)
    return flattened


def load_gold_triples(dataset: str, base_dir: Path | None = None, *, raw_data: Dict[str, Iterable] | None = None) -> TripleDict:
    """Load and flatten the gold triples for a dataset.

    The function expects files named ``{dataset}_processed.json`` unless
    ``raw_data`` is supplied directly. For BenchIE the JSON stores clusters of
    synonym relations per sentence; we take only the first triple from each
    cluster.
    """
    if raw_data is None:
        base = base_dir or Path('.')
        path = base / f"{dataset}_processed.json"
        with path.open('r', encoding='utf-8') as f:
            raw_data = json.load(f)

    gold: TripleDict = {}
    for sentence, entry in (raw_data or {}).items():
        flattened = _flatten_gold_entry(entry) if isinstance(entry, Iterable) else []
        gold[_sanitize_field(sentence)] = flattened
    return gold


def normalize_prediction_triples(
    predictions: Dict[str, Sequence[Sequence]], first_only: bool = False
) -> TripleDict:
    """Normalise predicted triples to ``{sentence: [[field, ...], ...]}``.

    Each extraction may have 2 or more string fields (subject, predicate,
    and optional object(s)).  Trailing confidence scores are stripped.
    """
    normalised: TripleDict = {}
    for sentence, triples in predictions.items():
        cleaned_sentence = _sanitize_field(sentence)
        normalised[cleaned_sentence] = []
        for triple in triples:
            normalised_triple = _normalize_triple(triple)
            if normalised_triple:
                normalised[cleaned_sentence].append(normalised_triple)
                if first_only:
                    break
    return normalised


def triple_to_extraction_text(triple: Sequence[Any]) -> str:
    """Convert one triple to plain extraction text.

    Confidence tails are removed and remaining fields are whitespace-normalized,
    then joined with a single space.
    """
    normalized = _normalize_triple(triple)
    if not normalized:
        return ""
    return " ".join(normalized).strip()


def triples_to_extraction_texts(triples: Sequence[Sequence[Any]]) -> List[str]:
    """Convert a sequence of triples to extraction texts."""
    texts: List[str] = []
    for triple in triples or []:
        text = triple_to_extraction_text(triple)
        if text:
            texts.append(text)
    return texts


def combine_extraction_texts(extractions: Sequence[str]) -> str:
    """Combine extraction strings into one text blob used by metrics.

    Output format mirrors historical behavior: `"e1. e2."`.
    """
    pieces = [e.strip() for e in (extractions or []) if isinstance(e, str) and e.strip()]
    if not pieces:
        return ""
    return "".join(f"{p}. " for p in pieces).strip()


def write_triples_to_tsv(triples: TripleDict, destination: Path, is_prediction: bool = False) -> None:
    """Serialise triples into the TSV layout expected by entailment_eval.py."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('w', encoding='utf-8') as f:
        for sentence, triple_list in triples.items():
            if not triple_list:
                continue
            for triple in triple_list:
                if len(triple) >= 3:
                    head, relation = triple[0], triple[1]
                    tail_parts = triple[2:]
                elif len(triple) == 2:
                    head, relation = triple[0], triple[1]
                    tail_parts = ['']
                else:
                    continue
                if is_prediction:
                    row = [sentence, 'pred', relation, head] + tail_parts
                else:
                    row = [sentence, relation, head] + tail_parts
                f.write('\t'.join(row) + '\n')


__all__ = [
    'combine_extraction_texts',
    'load_gold_triples',
    'normalize_prediction_triples',
    'triple_to_extraction_text',
    'triples_to_extraction_texts',
    'write_triples_to_tsv',
]
