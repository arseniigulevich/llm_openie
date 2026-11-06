from __future__ import annotations

import json
import os
from functools import reduce
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


_ELE_EMB_DICT: Optional[Dict[str, Any]] = None
_EMBEDDINGS_PATH: Optional[Path] = None


def save_embeddings_atomic(path: str | Path, emb_dict: Dict[str, Any]) -> None:
    """Write the embeddings dict so an interrupted write can't corrupt the file.

    The dict is multi-GB; a plain ``json.dump`` truncates the target if the
    process is killed (e.g. by a timeout) mid-write, leaving invalid JSON that
    breaks every later run. Writing to a sibling temp file and ``os.replace``-ing
    it into place makes the swap atomic: on failure the original file is untouched.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(emb_dict, f)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _default_embeddings_path() -> Path:
    # Workspace layout: OpenIE_evaluation/source/embeddings.py
    # Embeddings are stored in OpenIE_evaluation/uniqueness/gre_element_embedding_dict.json
    return Path(__file__).resolve().parents[1] / "uniqueness" / "gre_element_embedding_dict_hu.json"


def set_embeddings_path(path: str | Path) -> None:
    """Set the embeddings path for subsequent load_element_embeddings calls.
    
    This clears any cached embeddings so the new path will be used.
    """
    global _EMBEDDINGS_PATH, _ELE_EMB_DICT
    _EMBEDDINGS_PATH = Path(path) if path else None
    _ELE_EMB_DICT = None  # Clear cache so new path is used


def load_element_embeddings(*, verbose: bool = True) -> Dict[str, Any]:
    """Load and cache element embeddings.

    Uses a process-local singleton cache. Path priority:
    1. Path set via set_embeddings_path()
    2. OPENIE_EVAL_ELEM_EMB_PATH environment variable
    3. Default path
    """

    global _ELE_EMB_DICT
    if _ELE_EMB_DICT is not None:
        return _ELE_EMB_DICT

    if _EMBEDDINGS_PATH is not None:
        emb_path = _EMBEDDINGS_PATH
    else:
        emb_path = Path(os.environ.get("OPENIE_EVAL_ELEM_EMB_PATH", str(_default_embeddings_path())))
    if verbose:
        print("Loading element embeddings...")

    with emb_path.open("r", encoding="utf-8") as f:
        _ELE_EMB_DICT = json.load(f)

    return _ELE_EMB_DICT


def _strip_confidence(elements: list) -> list:
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


def get_triple_embedding(triple: List[str], emb_dict: Optional[Dict[str, Any]] = None) -> List[float]:
    """Compute an embedding for a variable-length extraction tuple.

    The tuple is expected to have the form ``[subj, pred, obj1, obj2, ...]``
    with an optional trailing confidence that is stripped automatically.
    The embedding is ``sum(entity embeddings) + predicate embedding`` where
    entity embeddings are all arguments except the predicate (index 1).
    """
    d = emb_dict if emb_dict is not None else load_element_embeddings()
    parts = _strip_confidence(list(triple))
    if len(parts) < 2:
        raise ValueError(f"Triple too short after stripping confidence: {triple}")
    pred_emb = np.array(d[parts[1]])
    entity_parts = [parts[0]] + parts[2:]  # subject + objects
    entity_emb = reduce(np.add, [np.array(d[e]) for e in entity_parts])
    return np.add(entity_emb, pred_emb).tolist()


def clear_embeddings_cache() -> None:
    """Free the in-process embeddings cache (call between datasets)."""
    global _ELE_EMB_DICT
    _ELE_EMB_DICT = None


def _collect_strings_from_obj(obj: Any, target: set) -> None:
    """Recursively collect non-empty strings from nested lists."""
    if isinstance(obj, str):
        s = obj.strip()
        if s:
            target.add(s)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _collect_strings_from_obj(item, target)


def fetch_missing_embeddings(
    strings: Iterable[str],
    emb_dict: Dict[str, Any],
    *,
    endpoint: str,
    model: str = "llm5",
    chunk_size: int = 64,
    verbose: bool = True,
) -> int:
    """Fetch embeddings for strings absent from emb_dict, updating it in-place.

    Sends requests in batches of `chunk_size` to avoid hitting request-size
    limits.  Strings whose embedding cannot be fetched are silently skipped.

    Returns the number of newly added embeddings.
    """
    from openai import OpenAI  # lazy import — only needed when API is configured

    missing = [s for s in strings if s and s not in emb_dict]
    if not missing:
        return 0

    if verbose:
        print(f"Fetching {len(missing)} missing embeddings from {endpoint} ...")

    client = OpenAI(base_url=endpoint.rstrip("/"), api_key="not-needed")
    added = 0

    for start in range(0, len(missing), chunk_size):
        batch = missing[start:start + chunk_size]
        try:
            response = client.embeddings.create(input=batch, model=model)
            for item in response.data:
                emb_dict[batch[item.index]] = item.embedding
                added += 1
        except Exception as exc:
            if verbose:
                print(f"  Warning: embedding fetch failed for chunk starting at {start}: {exc}")

    if verbose and added > 0:
        print(f"  Fetched {added}/{len(missing)} embeddings successfully.")

    return added


def prefetch_dataset_embeddings(
    extraction_files: List[Path],
    gt_data: Optional[Dict],
    emb_dict: Dict[str, Any],
    *,
    endpoint: str,
    model: str = "llm5",
    chunk_size: int = 64,
    save_path: Optional[Path] = None,
) -> int:
    """Pre-fetch all missing embeddings for a dataset before metric computation.

    Scans all system extraction files and the ground-truth mapping, collects
    every unique element string that is absent from `emb_dict`, fetches them
    in bulk from the embedding API, and optionally saves the updated dict
    back to `save_path` so future runs skip the API call entirely.

    Returns the total number of newly fetched embeddings.
    """
    all_strings: set = set()

    for path in extraction_files:
        if not Path(path).exists():
            continue
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            _collect_strings_from_obj(list(data.values()), all_strings)
        except Exception:
            continue

    if gt_data:
        _collect_strings_from_obj(list(gt_data.values()), all_strings)

    added = fetch_missing_embeddings(
        all_strings, emb_dict,
        endpoint=endpoint, model=model,
        chunk_size=chunk_size,
    )

    if added > 0 and save_path is not None:
        try:
            save_embeddings_atomic(save_path, emb_dict)
            print(f"  Saved updated embeddings to {save_path}")
        except Exception as exc:
            print(f"  Warning: could not save embeddings to {save_path}: {exc}")

    return added
