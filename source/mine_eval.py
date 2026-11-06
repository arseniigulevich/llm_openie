"""MINE evaluation, adapted from the MINE experiment of KGGen (kg-gen/experiments/MINE).

Works only on the MINE dataset: it needs the articles and their facts below and
graph-form extractions. evaluate.py calls it only for a dataset with ``mine = true``.

The MINE benchmark (Mo et al., 2025) measures *knowledge retention*: how much of
an article's information survives into a knowledge graph built from a system's
extractions. For each article it:

1. builds a KGGen ``Graph`` from the system's (subject, predicate, object) triples,
2. resolves duplicate entities/edges (``deduplicate`` method ``FULL``),
3. embeds the graph nodes and, for each of the article's known facts, retrieves a
   2-hop subgraph around the most similar nodes,
4. asks an LLM judge whether that retrieved context contains the fact.

The per-article MINE accuracy is the fraction of facts judged present; the global
score is the mean over articles. An article without facts (the dataset contains one
placeholder, id 106) gets no accuracy and is left out of the mean.

Data (under ``data/MINE``):
- ``mine_dataset_full.json`` — list of 101 articles, each with ``id``,
  ``essay_topic``, ``essay_content`` and ``generated_queries`` (15 facts).
- ``{system}_mine.json`` — dict keyed by article id (``"1"``..``"101"``); each value
  is a graph dict ``{essay_topic, entities, edges, relations}`` where ``relations``
  is a list of ``[subject, predicate, object]``.

This module only depends on the bundled minimal ``kg_gen`` package (``kg_gen/`` in
this folder) and on :mod:`source.hu_chat` for the LLM judge.
"""

from __future__ import annotations

import asyncio
import atexit
import json
import re
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from source.hu_chat import hu_instruct_async, DEFAULT_TEMPERATURE
from evaluation.concurrency import gather_bounded

# the minimal kg_gen subset bundled in kg_gen/ (run from cleaned_up/)
from kg_gen import KGGen
from kg_gen.models import Graph
from kg_gen.steps._3_deduplicate import DeduplicateMethod

_REPO_ROOT = Path(__file__).resolve().parent.parent
_JUDGE_PROMPT_PATH = _REPO_ROOT / "prompts" / "mine_fact_check.txt"

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_RETRIEVE_K = 8

# All KGGen / DSPy work must run on ONE persistent thread for the whole process.
# DSPy fixes its `settings` owner to the first thread that configures it and
# refuses changes from any other thread, even after the original thread dies. A
# fresh per-call pool would therefore work for the first system and fail for every
# subsequent one ("dspy.settings can only be changed by the thread that initially
# configured it"). A single shared, long-lived worker thread avoids that.
_DSPY_EXECUTOR: Optional[ThreadPoolExecutor] = None
_KGGEN_CACHE: dict = {}


def _dspy_executor() -> ThreadPoolExecutor:
    global _DSPY_EXECUTOR
    if _DSPY_EXECUTOR is None:
        _DSPY_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mine-dspy")
        atexit.register(_DSPY_EXECUTOR.shutdown, wait=False)
    return _DSPY_EXECUTOR


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_mine_dataset(data_dir: str | Path) -> Dict[str, Dict[str, Any]]:
    """Load the MINE article dataset, keyed by article id (str).

    Returns ``{id: {"essay_topic", "essay_content", "facts": [str, ...]}}``.
    """
    path = Path(data_dir) / "mine_dataset_full.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, Dict[str, Any]] = {}
    for item in raw:
        out[str(item["id"])] = {
            "essay_topic": item.get("essay_topic", ""),
            "essay_content": item.get("essay_content", ""),
            "facts": list(item.get("generated_queries", [])),
        }
    return out


def load_mine_extractions(path: str | Path) -> Dict[str, Dict[str, Any]]:
    """Load a ``{system}_mine.json`` file: ``{id: {entities, edges, relations}}``."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def mine_extractions_to_text_triples(
    extractions_by_id: Dict[str, Dict[str, Any]],
    dataset: Dict[str, Dict[str, Any]],
) -> Dict[str, List[List[str]]]:
    """Convert MINE graph extractions into the standard ``{text: [[s,p,o], ...]}``.

    The "text" key is each article's ``essay_content`` so that the generic
    evaluation metrics (FS, GS, SS, US, Coherence, sentence entailment, ...) can
    run on the MINE dataset unchanged. Articles missing from ``dataset`` are
    skipped.
    """
    out: Dict[str, List[List[str]]] = {}
    for art_id, entry in extractions_by_id.items():
        meta = dataset.get(str(art_id))
        if not meta:
            continue
        text = meta["essay_content"]
        relations = entry.get("relations", []) or []
        triples = [list(r[:3]) for r in relations if isinstance(r, (list, tuple)) and len(r) >= 3]
        out[text] = triples
    return out


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph(entry: Dict[str, Any]) -> Graph:
    """Build a kg_gen ``Graph`` from a MINE extraction entry.

    Entities/edges are taken from the entry when present, otherwise derived from
    the relations so the graph is always self-consistent.
    """
    relations_raw = entry.get("relations", []) or []
    relations = {
        (str(r[0]).strip(), str(r[1]).strip(), str(r[2]).strip())
        for r in relations_raw
        if isinstance(r, (list, tuple)) and len(r) >= 3 and all(str(x).strip() for x in r[:3])
    }
    entities = {s for s, _, _ in relations} | {o for _, _, o in relations}
    edges = {p for _, p, _ in relations}
    # Merge any extra declared entities/edges (harmless supersets).
    entities |= {str(e).strip() for e in (entry.get("entities") or []) if str(e).strip()}
    edges |= {str(e).strip() for e in (entry.get("edges") or []) if str(e).strip()}
    return Graph(entities=entities, edges=edges, relations=relations)


# ---------------------------------------------------------------------------
# KGGen factory (configured against the project's OpenAI-compatible endpoint)
# ---------------------------------------------------------------------------

def build_kggen(
    endpoint: Optional[str],
    model: Optional[str] = None,
    *,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    litellm_provider: str = "hosted_vllm",
) -> KGGen:
    """Construct a KGGen configured to use the project's LLM endpoint for the
    FULL-deduplication LLM clustering, and ``embedding_model`` for retrieval.

    The endpoint is OpenAI-compatible (vLLM). We prefix the model with a litellm
    provider that routes to ``/chat/completions`` (``hosted_vllm`` by default);
    we avoid the ``openai/`` prefix because KGGen would then use the unsupported
    Responses API for it.
    """
    resolved = model
    if resolved is None and endpoint is not None:
        from source.hu_chat import _resolve_model
        resolved = _resolve_model(endpoint)
    resolved = resolved or "gpt-4o"
    litellm_model = resolved if "/" in resolved else f"{litellm_provider}/{resolved}"

    return KGGen(
        model=litellm_model,
        api_base=endpoint,
        api_key="not-used",
        temperature=0.0,
        retrieval_model=embedding_model,
    )


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

_JUDGE_PROMPT_CACHE: Optional[str] = None


def _judge_prompt() -> str:
    global _JUDGE_PROMPT_CACHE
    if _JUDGE_PROMPT_CACHE is None:
        _JUDGE_PROMPT_CACHE = _JUDGE_PROMPT_PATH.read_text(encoding="utf-8")
    return _JUDGE_PROMPT_CACHE


def _parse_judge(response: str) -> int:
    """Parse the judge response into 1 (present) or 0 (absent)."""
    text = (response or "").strip().lower()
    m = re.search(r"[01]", text)
    if m:
        return int(m.group(0))
    if any(w in text for w in ("yes", "true", "present", "contained", "entail")):
        return 1
    return 0


async def _judge_fact_async(
    fact: str,
    context: str,
    endpoint: Optional[str],
    model: Optional[str],
    semaphore: Optional[asyncio.Semaphore],
    temperature: float = DEFAULT_TEMPERATURE,
) -> int:
    prompt = _judge_prompt().replace("$CONTEXT$", context or "").replace("$FACT$", fact)
    raw = await hu_instruct_async(prompt, endpoint, model, semaphore, max_tokens=8, temperature=temperature)
    return _parse_judge(raw)


# ---------------------------------------------------------------------------
# Per-article retrieval (sync, runs in a worker thread)
# ---------------------------------------------------------------------------

def _retrieve_contexts_for_article(
    kggen: KGGen,
    entry: Dict[str, Any],
    facts: List[str],
    *,
    dedup_method: DeduplicateMethod = DeduplicateMethod.FULL,
    k: int = DEFAULT_RETRIEVE_K,
) -> List[str]:
    """Build + dedup the graph, embed nodes, and return the retrieved context
    string for each fact. Synchronous; call via ``asyncio.to_thread``.
    """
    graph = build_graph(entry)
    if not graph.relations:
        return ["" for _ in facts]

    deduped = kggen.deduplicate(graph, method=dedup_method)
    nx_graph = kggen.to_nx(deduped)
    if nx_graph.number_of_nodes() == 0:
        return ["" for _ in facts]
    node_embeddings, _ = kggen.generate_embeddings(nx_graph)

    contexts: List[str] = []
    for fact in facts:
        try:
            *_, context_text = kggen.retrieve(fact, node_embeddings, nx_graph, k=k)
        except Exception:
            context_text = ""
        contexts.append(context_text)
    return contexts


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def calculate_mine_accuracy_async(
    extractions_by_id: Dict[str, Dict[str, Any]],
    dataset: Dict[str, Dict[str, Any]],
    endpoint: Optional[str],
    model: Optional[str] = None,
    semaphore: Optional[asyncio.Semaphore] = None,
    *,
    kggen: Optional[KGGen] = None,
    dedup_method: DeduplicateMethod = DeduplicateMethod.FULL,
    k: int = DEFAULT_RETRIEVE_K,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    article_ids: Optional[List[str]] = None,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Tuple[float, Dict[str, Dict[str, Any]]]:
    """Compute MINE accuracy of a system's extractions.

    Returns ``(global_accuracy, per_article)`` where ``per_article[id]`` is
    ``{"accuracy", "essay_topic", "per_fact": [{"fact", "evaluation", "context"}]}``.

    Graph construction, FULL deduplication, embedding and retrieval run
    **sequentially** on a single dedicated worker thread. KGGen's LLM clustering
    calls ``dspy.configure``, and DSPy only permits the thread that first
    configured its settings to change them; pinning all KGGen work to one thread
    (rather than ``asyncio.to_thread``'s rotating pool) avoids the cross-thread
    error. The LLM fact judgements run **concurrently**, bounded by ``semaphore``.
    """
    loop = asyncio.get_running_loop()
    dspy_pool = _dspy_executor()  # shared, persistent single worker (see note above)

    if kggen is None:
        cache_key = (endpoint, model, embedding_model)
        kggen = _KGGEN_CACHE.get(cache_key)
        if kggen is None:
            kggen = await loop.run_in_executor(
                dspy_pool, partial(build_kggen, endpoint, model, embedding_model=embedding_model)
            )
            _KGGEN_CACHE[cache_key] = kggen

    ids = article_ids if article_ids is not None else list(extractions_by_id.keys())
    ids = [str(i) for i in ids if str(i) in dataset and str(i) in extractions_by_id]

    # Phase A (the shared dedicated thread): build/dedup/embed/retrieve per article.
    contexts_by_id: Dict[str, List[str]] = {}
    facts_by_id: Dict[str, List[str]] = {}
    for art_id in ids:
        facts = dataset[art_id]["facts"]
        facts_by_id[art_id] = facts
        if not facts:
            contexts_by_id[art_id] = []
            continue
        contexts_by_id[art_id] = await loop.run_in_executor(
            dspy_pool,
            partial(
                _retrieve_contexts_for_article,
                kggen,
                extractions_by_id[art_id],
                facts,
                dedup_method=dedup_method,
                k=k,
            ),
        )

    # Phase B (concurrent): judge every (article, fact) pair.
    factories = []
    flat_meta: List[Tuple[str, int]] = []
    for art_id in ids:
        facts = facts_by_id[art_id]
        contexts = contexts_by_id[art_id]
        for fi, fact in enumerate(facts):
            ctx = contexts[fi] if fi < len(contexts) else ""
            factories.append(partial(_judge_fact_async, fact, ctx, endpoint, model, semaphore,
                                     temperature=temperature))
            flat_meta.append((art_id, fi))

    # On a failed judgement, the other requests are cancelled instead of left running.
    evaluations = await gather_bounded(factories, desc="MINE judge")

    # Reassemble per-article results.
    per_article: Dict[str, Dict[str, Any]] = {}
    for art_id in ids:
        per_article[art_id] = {
            "essay_topic": dataset[art_id].get("essay_topic", ""),
            "accuracy": None,
            "per_fact": [
                {"fact": f, "evaluation": None, "context": ""}
                for f in facts_by_id[art_id]
            ],
        }
    for (art_id, fi), ev in zip(flat_meta, evaluations):
        entry = per_article[art_id]["per_fact"][fi]
        entry["evaluation"] = int(ev)
        entry["context"] = contexts_by_id[art_id][fi] if fi < len(contexts_by_id[art_id]) else ""

    article_accuracies: List[float] = []
    for art_id in ids:
        if not facts_by_id[art_id]:
            # An article without reference facts cannot be scored (mine_dataset_full.json
            # contains one such placeholder, id 106, with empty text). Counting it as 0 would
            # penalise every system that has an entry for it; leave it out of the mean.
            per_article[art_id]["accuracy"] = None
            continue
        evals = [pf["evaluation"] for pf in per_article[art_id]["per_fact"] if pf["evaluation"] is not None]
        acc = (sum(evals) / len(evals)) if evals else 0.0
        per_article[art_id]["accuracy"] = acc
        article_accuracies.append(acc)

    global_accuracy = (sum(article_accuracies) / len(article_accuracies)) if article_accuracies else 0.0
    return global_accuracy, per_article
