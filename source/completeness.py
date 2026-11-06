"""Completeness Score: CS-r (GenRES) and CS-p (BenchIE-style precision).

Both directions compare predicted triples with gold *units* by the cosine
similarity of their embeddings. A gold unit is a fact synset for BenchIE-style
gold (a list of paraphrase clusters) and a single gold triple for flat gold.
A prediction matches a unit when it reaches the threshold with any member of
the unit.

    CS-r = covered units / all units                          (Jiang et al. 2024)
    CS-p = covered units / (covered units + unmatched predictions)
                                                              (Gashteovski et al. 2022)

CS-p follows BenchIE's precision: a unit is rewarded once however many
predictions match it, and only predictions that match no unit count against
the system. Redundant predictions therefore neither help nor hurt CS-p; the
Uniqueness Score measures redundancy.

Edge cases, identical for both directions:
  - no predictions for a text: CS-r = CS-p = 0;
  - no gold units for a text (empty gold, or no unit that can be embedded):
    undefined, the text is left out of the per-text scores and the mean;
  - text not in the gold file: left out.

A triple is embedded from its non-empty slots, so an intransitive triple such as
("He", "died", "") keeps its subject and predicate instead of failing. A
predicted triple that still cannot be embedded counts as unmatched; a gold
triple that cannot be embedded is dropped from its unit.
"""
import json
import os
from collections import defaultdict
from functools import reduce

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from source.embeddings import load_element_embeddings, _strip_confidence

gt_triple_emb_store = {}


def _ensure_embeddings_loaded():
    return load_element_embeddings()


def clear_gt_caches() -> None:
    """Clear per-dataset ground-truth embedding caches between datasets."""
    global gt_triple_emb_store
    gt_triple_emb_store = {}


def _embed_triple(triple, emb_dict):
    """Embedding of a triple from its non-empty slots, or None.

    Same composition as source.embeddings.get_triple_embedding (sum of the
    argument embeddings plus the predicate embedding), except that empty
    argument slots are skipped. Subject and predicate must be present.
    """
    try:
        parts = _strip_confidence(list(triple))
        if len(parts) < 2:
            return None
        subj, pred = parts[0], parts[1]
        if not (isinstance(subj, str) and subj.strip() and isinstance(pred, str) and pred.strip()):
            return None
        args = [subj] + [e for e in parts[2:] if isinstance(e, str) and e.strip()]
        entity_emb = reduce(np.add, [np.array(emb_dict[e]) for e in args])
        return np.add(entity_emb, np.array(emb_dict[pred]))
    except (KeyError, TypeError, ValueError):
        return None


def _is_clustered(entry) -> bool:
    """True for BenchIE-style gold: a list of clusters, each a list of triples."""
    return (isinstance(entry, list) and len(entry) > 0 and isinstance(entry[0], list)
            and len(entry[0]) > 0 and isinstance(entry[0][0], list))


def _gold_units(entry, emb_dict):
    """Embedded gold units for one text: a list of (n_members, dim) arrays."""
    if _is_clustered(entry):
        clusters = entry
    else:
        # Identical flat gold triples are one fact.
        unique = {str(t): t for t in (entry or [])}
        clusters = [[t] for t in unique.values()]
    units = []
    for cluster in clusters:
        members = []
        for triple in cluster:
            key = str(triple)
            if key not in gt_triple_emb_store:
                gt_triple_emb_store[key] = _embed_triple(triple, emb_dict)
            if gt_triple_emb_store[key] is not None:
                members.append(gt_triple_emb_store[key])
        if members:
            units.append(np.array(members))
    return units


def _load_gold(dataset, gt_text_triples):
    if gt_text_triples is not None:
        return gt_text_triples
    candidate_paths = [
        os.path.join('..', 'datasets', 'processed', f'{dataset}_processed.json'),
        f'{dataset}_processed.json',
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
    raise FileNotFoundError(
        f"Unable to locate ground-truth file for dataset '{dataset}'. Tried: {candidate_paths}"
    )


def _score_texts(data_to_evaluate, gt_text_triples, threshold):
    """Per-text recall, precision and match details."""
    emb_dict = _ensure_embeddings_loaded()
    scores = {}
    for text, triples in data_to_evaluate.items():
        if text not in gt_text_triples:
            continue
        units = _gold_units(gt_text_triples[text], emb_dict)
        if not units:
            continue
        preds = triples if isinstance(triples, list) else []
        if not preds:
            scores[text] = {'r': 0.0, 'p': 0.0, 'num_units': len(units), 'num_predictions': 0,
                            'covered': 0, 'unmatched_predictions': 0,
                            'unit_best_scores': [0.0] * len(units), 'pred_best_scores': []}
            continue

        pred_embs = [_embed_triple(t, emb_dict) for t in preds]
        valid = [i for i, e in enumerate(pred_embs) if e is not None]
        # best[i, u] = best similarity of prediction i to any member of unit u
        best = np.full((len(preds), len(units)), -1.0)
        if valid:
            P = np.array([pred_embs[i] for i in valid])
            for u, G in enumerate(units):
                best[valid, u] = cosine_similarity(P, G).max(axis=1)
        match = best >= threshold
        covered = int(match.any(axis=0).sum())
        unmatched = int((~match.any(axis=1)).sum())
        scores[text] = {
            'r': covered / len(units),
            'p': covered / (covered + unmatched) if covered + unmatched else 0.0,
            'num_units': len(units),
            'num_predictions': len(preds),
            'covered': covered,
            'unmatched_predictions': unmatched,
            'unit_best_scores': best.max(axis=0).tolist(),
            'pred_best_scores': best.max(axis=1).tolist(),
        }
    return scores


def _aggregate(scores, key, output_all_scores):
    text2cs = {text: s[key] for text, s in scores.items()}
    details = defaultdict(dict)
    for text, s in scores.items():
        details[text] = {k: v for k, v in s.items() if k not in ('r', 'p')}
    avg = float(np.mean(list(text2cs.values()))) if text2cs else 0.0
    if output_all_scores:
        return avg, details, text2cs
    return avg, details


def calculate_completeness_score(
    data_to_evaluate,
    dataset,
    model_name=None,
    threshold=0.95,
    output_all_scores=False,
    *,
    gt_text_triples=None,
):
    """CS-r: share of gold units covered by at least one predicted triple.

    Args:
        data_to_evaluate: Mapping from text to predicted triples.
        dataset: Dataset name, used to locate the gold file when
            ``gt_text_triples`` is not given.
        model_name: Unused; kept for API compatibility.
        threshold: Cosine similarity at which a prediction matches a unit.
        output_all_scores: If true, also return the per-text scores.
        gt_text_triples: Preloaded gold mapping (flat or BenchIE-style).

    Returns:
        (mean CS-r, details) or (mean CS-r, details, per-text CS-r).
    """
    gold = _load_gold(dataset, gt_text_triples)
    return _aggregate(_score_texts(data_to_evaluate, gold, threshold), 'r', output_all_scores)


def calculate_completeness_precision_score(
    data_to_evaluate,
    dataset,
    model_name=None,
    threshold=0.95,
    output_all_scores=False,
    *,
    gt_text_triples=None,
):
    """CS-p: covered gold units / (covered gold units + unmatched predictions).

    Same arguments and return layout as :func:`calculate_completeness_score`.
    """
    gold = _load_gold(dataset, gt_text_triples)
    return _aggregate(_score_texts(data_to_evaluate, gold, threshold), 'p', output_all_scores)
