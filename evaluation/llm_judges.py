"""The LLM-judged scores: factualness (FS), granularity (GS) and coherence.

Each score sends one prompt per triple to the chat endpoint; the requests of all
systems share one semaphore, which bounds the requests in flight.

Empty extraction sets are treated as in the thesis runs, which differs per score:
FS and GS give a text without triples no score (None) and leave it out of the mean;
Coherence gives it 0.0 and includes it in the mean.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections import defaultdict
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from evaluation.concurrency import DEFAULT_WINDOW, gather_bounded
from source.hu_chat import DEFAULT_TEMPERATURE, hu_instruct_async

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
MAX_TOKENS = 500


@lru_cache(maxsize=None)
def load_prompt(filename: str) -> str:
    return (PROMPTS_DIR / filename).read_text(encoding="utf-8").strip()


_STRIP = "`\"' \t\n\r.,:;!?()[]{}"


def parse_true_false(response: Any) -> bool:
    """FS label: an exact or first ``true``/``false`` token, else ``yes``/``no``, else False."""
    text = str(response or "").strip().lower()
    if not text:
        return False
    stripped = text.strip(_STRIP)
    if stripped in ("true", "false"):
        return stripped == "true"
    match = re.search(r"\b(true|false)\b", text)
    if match:
        return match.group(1) == "true"
    match = re.search(r"\b(yes|no)\b", text)
    if match:
        return match.group(1) == "yes"
    return False


def parse_coherence(response: Any) -> bool:
    """Coherence label; ``incoherent`` is tested first because it contains ``coherent``."""
    text = str(response or "").strip().lower()
    if not text:
        return False
    stripped = text.strip(_STRIP)
    if stripped in ("coherent", "incoherent"):
        return stripped == "coherent"
    if re.search(r"\bincoherent\b", text):
        return False
    return bool(re.search(r"\bcoherent\b", text))


def granularity_from_response(response: str) -> float:
    """GS of one triple: exp(-n), n = number of bracket pairs in the judge's answer,
    where a single pair (the triple itself, no split) counts as 0."""
    n = min(response.count("["), response.count("]"))
    if n == 1:
        n = 0
    return float(np.exp(-n))


class Judge:
    """Sends judge prompts to the endpoint with the run's settings."""

    def __init__(self, endpoint: Optional[str], model: Optional[str],
                 semaphore: Optional[asyncio.Semaphore],
                 temperature: float = DEFAULT_TEMPERATURE, window: int = DEFAULT_WINDOW):
        self.endpoint = endpoint
        self.model = model
        self.semaphore = semaphore
        self.temperature = temperature
        self.window = window

    async def ask_all(self, prompts: List[str], desc: str) -> List[str]:
        factories = [
            partial(hu_instruct_async, p, self.endpoint, self.model, self.semaphore,
                    max_tokens=MAX_TOKENS, temperature=self.temperature)
            for p in prompts
        ]
        return await gather_bounded(factories, window=self.window, desc=desc)


def _items(data: Dict[str, List]) -> List[Tuple[str, Any]]:
    return [(text, triple) for text, triples in data.items() for triple in triples]


async def factualness(data: Dict[str, List], judge: Judge) -> Tuple[float, Dict[str, Dict]]:
    """FS: share of triples the judge finds supported by the source text."""
    template = load_prompt("fact_checker.txt")
    items = _items(data)
    prompts = [template.replace("$TEXT$", text).replace("$TRIPLE$", json.dumps(triple))
               for text, triple in items]
    responses = await judge.ask_all(prompts, "FS")
    labels: Dict[str, List[bool]] = defaultdict(list)
    for (text, _), response in zip(items, responses):
        labels[text].append(parse_true_false(response))
    details = {text: {"score": sum(ls) / len(ls), "results": ls} for text, ls in labels.items()}
    mean = float(np.mean([d["score"] for d in details.values()])) if details else 0.0
    return mean, details


async def coherence(data: Dict[str, List], judge: Judge) -> Tuple[float, Dict[str, Dict]]:
    """Coherence: share of triples the judge finds interpretable without the source."""
    template = load_prompt("coherence_checker.txt")
    items = _items(data)
    prompts = [template.replace("$TRIPLE$", json.dumps(triple)) for _, triple in items]
    responses = await judge.ask_all(prompts, "Coherence")
    labels: Dict[str, List[bool]] = defaultdict(list)
    for (text, _), response in zip(items, responses):
        labels[text].append(parse_coherence(response))
    details = {}
    for text in data:
        ls = labels.get(text, [])
        details[text] = {"score": float(sum(ls) / len(ls)) if ls else 0.0, "results": [bool(v) for v in ls]}
    mean = float(np.mean([d["score"] for d in details.values()])) if details else 0.0
    return mean, details


async def granularity(data: Dict[str, List], judge: Judge) -> Tuple[float, Dict[str, Dict[str, str]], Dict[str, float]]:
    """GS: mean exp(-n_sub) over the triples of a text. Returns the mean over texts, the
    raw answers per text and triple, and the per-text scores."""
    template = load_prompt("granularity_checker.txt")
    items = [(text, json.dumps(triple)) for text, triple in _items(data)]
    prompts = [template.replace("$TRIPLE$", triple_json) for _, triple_json in items]
    responses = await judge.ask_all(prompts, "GS")
    answers: Dict[str, Dict[str, str]] = defaultdict(dict)
    for (text, triple_json), response in zip(items, responses):
        answers[text][triple_json] = response.strip() if response else ""
    text2gs = {text: float(np.mean([granularity_from_response(r) for r in by_triple.values()]))
               for text, by_triple in answers.items()}
    mean = float(np.mean(list(text2gs.values()))) if text2gs else 1.0
    return mean, dict(answers), text2gs
