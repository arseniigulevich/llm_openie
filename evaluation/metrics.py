"""The metrics of the runner as a registry.

Each ``Metric`` names its config switch, the result keys it writes, the phase it runs
in, and an async ``compute`` that returns dataset-level values and per-sentence values.
The runner (``evaluation.runner``) handles everything common to all metrics: the
config switch, resuming, recompute flags, saving, and the run-summary status.

Phases: ``main`` runs all metrics except surprisal; ``ss`` runs surprisal after the
entailment cross-encoder has been unloaded, so the two models never share the GPU.
"""

from __future__ import annotations

import asyncio
import pickle
import tempfile
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

import numpy as np

from eval_config import DatasetConfig, EvalConfig
from evaluation import llm_judges
from evaluation.gpu import release_cuda_memory
from evaluation.inputs import SystemInputs
from evaluation.results import ResultStore, to_serializable
from source import entailment_eval
from source.converter import _sanitize_field, load_gold_triples, write_triples_to_tsv

MAIN, SS = "main", "ss"
PHASES = (MAIN, SS)

PerSentence = Dict[str, Dict[str, Any]]


@dataclass
class MetricOutput:
    dataset: Dict[str, Any]
    per_sentence: PerSentence = field(default_factory=dict)


class NotApplicable(Exception):
    """The metric does not apply to this dataset (recorded as skipped, not failed)."""


class ModelManager:
    """Loads the entailment cross-encoder once, on first use, for all systems."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()

    async def cross_encoder(self):
        async with self._lock:
            if entailment_eval.sota_model is None:
                await asyncio.to_thread(entailment_eval.load_cross_encoder)
            return entailment_eval.sota_model

    @staticmethod
    def unload_cross_encoder() -> None:
        if entailment_eval.sota_model is not None:
            entailment_eval.unload_cross_encoder()
            release_cuda_memory()

    @staticmethod
    def unload_surprisal(model_name: str) -> None:
        from source import surprisal
        surprisal.unload_surprisal_model(model_name)
        release_cuda_memory()


@dataclass
class MetricContext:
    """What a metric's compute function sees for one system on one dataset."""
    config: EvalConfig
    dataset_config: DatasetConfig
    inputs: SystemInputs
    store: ResultStore
    executor: Any                      # concurrent.futures.Executor for blocking work
    judge: llm_judges.Judge
    models: ModelManager
    mine_max_articles: Optional[int] = None

    @property
    def predictions(self):
        return self.inputs.predictions

    @property
    def sentences(self):
        return self.inputs.sentences

    async def run_blocking(self, fn: Callable, *args, **kwargs):
        """Run a blocking (CPU, GPU or file) function in the thread pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, partial(fn, *args, **kwargs))


@dataclass(frozen=True)
class Metric:
    name: str                                   # config switch and recompute_<name>
    keys: Tuple[str, ...]                       # dataset- and per-sentence result keys
    compute: Callable[[MetricContext], Awaitable[MetricOutput]]
    phase: str = MAIN
    summary_names: Tuple[str, ...] = ()         # names in the run summary (default: keys)
    mine_only: bool = False
    needs_gold: bool = False
    is_done: Optional[Callable[[MetricContext], bool]] = None

    def labels(self) -> Tuple[str, ...]:
        return self.summary_names or self.keys

    def done(self, ctx: MetricContext) -> bool:
        if self.is_done is not None:
            return self.is_done(ctx)
        return ctx.store.has(self.keys, ctx.sentences)


def _opt_float(value) -> Optional[float]:
    return float(value) if value is not None else None


# --------------------------------------------------------------------------- TS, US, CS

async def _topical(ctx: MetricContext) -> MetricOutput:
    from source.topical import calculate_ts_score
    name = ctx.dataset_config.lda_model
    dictionary_path = ctx.config.topical_dir / f"{name}_dictionary.pkl"
    lda_path = ctx.config.topical_dir / f"{name}_lda.pkl"
    if not (name and dictionary_path.exists() and lda_path.exists()):
        raise FileNotFoundError("Missing LDA model or dictionary for TS")

    def _load(path: Path):
        with open(path, "rb") as f:
            return pickle.load(f)

    dictionary = await ctx.run_blocking(_load, dictionary_path)
    lda = await ctx.run_blocking(_load, lda_path)
    ts, details = await ctx.run_blocking(calculate_ts_score, ctx.predictions, dictionary, lda,
                                         output_all_scores=True)
    return MetricOutput(
        {"TS": float(ts), "TS_details": to_serializable(details)},
        {s: {"TS": float(details.get(s, 0.0))} for s in ctx.sentences},
    )


async def _uniqueness(ctx: MetricContext) -> MetricOutput:
    from source.uniqueness import calculate_uniqueness_score
    us, details = await ctx.run_blocking(calculate_uniqueness_score, ctx.predictions,
                                         output_all_scores=True)
    return MetricOutput(
        {"US": float(us), "US_details": to_serializable(details)},
        {s: {"US": float(details.get(s, 1.0))} for s in ctx.sentences},
    )


async def _completeness(ctx: MetricContext) -> MetricOutput:
    from source.completeness import calculate_completeness_precision_score, calculate_completeness_score
    name = ctx.dataset_config.name
    recall, r_details, text2r = await ctx.run_blocking(
        calculate_completeness_score, ctx.predictions, name,
        output_all_scores=True, gt_text_triples=ctx.inputs.gold_keyed)
    precision, p_details, text2p = await ctx.run_blocking(
        calculate_completeness_precision_score, ctx.predictions, name,
        output_all_scores=True, gt_text_triples=ctx.inputs.gold_keyed)
    return MetricOutput(
        {"CS-r": float(recall), "CS-p": float(precision),
         "CS-r_details": to_serializable(r_details), "CS-p_details": to_serializable(p_details)},
        {s: {"CS-r": _opt_float(text2r.get(s)), "CS-p": _opt_float(text2p.get(s))} for s in ctx.sentences},
    )


# --------------------------------------------------------------------------- Entailment

def _prf(scores) -> Optional[Dict[str, float]]:
    if scores is None:
        return None
    return {"precision": float(scores[0]), "recall": float(scores[1]), "f1": float(scores[2])}


def _entailment_blocking(predictions, gold_triples, model, prediction_name: str):
    """All three entailment granularities; only the sentence granularity without gold."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        prediction_file, stats_file = tmp / "prediction.tsv", tmp / "sent_level.tsv"
        write_triples_to_tsv(predictions, prediction_file, is_prediction=True)
        sent2pred = entailment_eval.read_extraction_file(str(prediction_file), 1)
        info: Dict[str, Any] = {}
        if gold_triples is None:
            # Synthetic gold keyed by the predicted sentences; the sentence granularity
            # compares the source with the joined predictions and does not read it.
            sent2gold = {s: [] for s in sent2pred}
            sentence = entailment_eval.calculate_sentence_tuple_score(
                sent2gold, sent2pred, f"{prediction_name}_sent2tuple", info, model, str(stats_file))
            return sentence, None, None, info
        gold_file = tmp / "gold.tsv"
        write_triples_to_tsv(gold_triples, gold_file)
        sent2gold = entailment_eval.read_extraction_file(str(gold_file), 0)
        sentence = entailment_eval.calculate_sentence_tuple_score(
            sent2gold, sent2pred, f"{prediction_name}_sent2tuple", info, model, str(stats_file))
        combined = entailment_eval.calculate_combined_tuple_tuple_score(
            sent2gold, sent2pred, f"{prediction_name}_combined_tuple2tuple", info, model, str(stats_file))
        tuple_ = entailment_eval.calculate_tuple_tuple_score(
            sent2gold, sent2pred, f"{prediction_name}_tuple2tuple", info, model, str(stats_file))
        return sentence, combined, tuple_, info


async def _entailment(ctx: MetricContext) -> MetricOutput:
    model = await ctx.models.cross_encoder()
    gold = ctx.inputs.gold
    gold_triples = load_gold_triples(ctx.dataset_config.name, raw_data=gold) if gold is not None else None
    if gold_triples is None:
        print(f"[{ctx.inputs.system}] Entailment: no gold, sentence granularity only")
    try:
        sentence, combined, tuple_, info = await ctx.run_blocking(
            _entailment_blocking, ctx.predictions, gold_triples, model,
            f"{ctx.dataset_config.name}_{ctx.inputs.system}")
    finally:
        release_cuda_memory()
    details = to_serializable(info)
    return MetricOutput(
        {"Entailment": {"sentence": _prf(sentence), "combined_tuple": _prf(combined), "tuple": _prf(tuple_)},
         "Entailment_sentence_details": details},
        {s: {"Entailment": details.get(s)} for s in ctx.sentences},
    )


# --------------------------------------------------------------------------- Surprisal

_SS_RAW = (("S-r_raw", "ss_recall_raw"), ("S-p_raw", "ss_precision_raw"),
           ("S-f1_raw", "ss_f1_raw"), ("SS-A", "atomicity"))


async def _surprisal(ctx: MetricContext) -> MetricOutput:
    from source.surprisal import calculate_surprisal_score
    recall, details, text2r, precision, f1, text2p, text2f1 = await ctx.run_blocking(
        calculate_surprisal_score, ctx.predictions, output_all_scores=True, model_name=ctx.config.ss_model)
    per_sentence: PerSentence = {}
    for s in ctx.sentences:
        entry = {"S-r": _opt_float(text2r.get(s)), "S-p": _opt_float(text2p.get(s)),
                 "S-f1": _opt_float(text2f1.get(s))}
        det = details.get(s) or {}
        for key, dkey in _SS_RAW:  # uncorrected scores and the copy-rate weight
            entry[key] = _opt_float(det.get(dkey))
        per_sentence[s] = entry
    dataset = {"S-r": float(recall), "S-p": float(precision), "S-f1": float(f1),
               "S-r_details": to_serializable(details)}
    for key in ("S-r_raw", "S-p_raw", "S-f1_raw"):
        vals = [per_sentence[s][key] for s in ctx.sentences
                if ctx.predictions.get(s) and per_sentence[s][key] is not None]
        dataset[key] = float(np.mean(vals)) if vals else 0.0
    return MetricOutput(dataset, per_sentence)


# --------------------------------------------------------------------------- LLM judges

async def _factualness(ctx: MetricContext) -> MetricOutput:
    fs, details = await llm_judges.factualness(ctx.predictions, ctx.judge)
    return MetricOutput(
        {"FS": float(fs), "FS_details": to_serializable(details)},
        {s: {"FS": _opt_float(details.get(s, {}).get("score"))} for s in ctx.sentences},
    )


async def _granularity(ctx: MetricContext) -> MetricOutput:
    gs, answers, text2gs = await llm_judges.granularity(ctx.predictions, ctx.judge)
    return MetricOutput(
        {"GS": float(gs), "GS_details": to_serializable(answers)},
        {s: {"GS": _opt_float(text2gs.get(s))} for s in ctx.sentences},
    )


async def _coherence(ctx: MetricContext) -> MetricOutput:
    coh, details = await llm_judges.coherence(ctx.predictions, ctx.judge)
    return MetricOutput(
        {"Coherence": float(coh), "Coherence_details": to_serializable(details)},
        {s: {"Coherence": _opt_float(details.get(s, {}).get("score"))} for s in ctx.sentences},
    )


# --------------------------------------------------------------------------- MINE

def mine_article_ids(ctx: MetricContext):
    ids = [str(i) for i in ctx.inputs.mine_extractions if str(i) in ctx.inputs.mine_dataset]
    return ids[:ctx.mine_max_articles] if ctx.mine_max_articles else ids


def _mine_done(ctx: MetricContext) -> bool:
    if "MINE" not in ctx.store.dataset:
        return False
    # A run limited to a few articles (--mine-max-articles) does not count as complete
    # for a later full run. Results without the count predate it and count as complete.
    scored = ctx.store.dataset.get("MINE_num_articles")
    return scored is None or scored >= len(mine_article_ids(ctx))


async def _mine(ctx: MetricContext) -> MetricOutput:
    from source import mine_eval
    ids = mine_article_ids(ctx)
    accuracy, per_article = await mine_eval.calculate_mine_accuracy_async(
        ctx.inputs.mine_extractions, ctx.inputs.mine_dataset,
        ctx.judge.endpoint, ctx.judge.model, ctx.judge.semaphore,
        article_ids=ids, temperature=ctx.judge.temperature,
    )
    per_sentence: PerSentence = {}
    for art_id, info in per_article.items():
        meta = ctx.inputs.mine_dataset.get(str(art_id))
        if meta:  # same sanitised key as the generic metrics' per_sentence entries
            per_sentence[_sanitize_field(meta["essay_content"])] = {"MINE": _opt_float(info.get("accuracy"))}
    return MetricOutput(
        {"MINE": float(accuracy), "MINE_details": to_serializable(per_article), "MINE_num_articles": len(ids)},
        per_sentence,
    )


def _entailment_done(ctx: MetricContext) -> bool:
    # Checks the per-sentence entries, which compact results keep, rather than
    # Entailment_sentence_details, which compact results drop.
    return ctx.store.has(("Entailment",), ctx.sentences)


METRICS: Tuple[Metric, ...] = (
    Metric("TS", ("TS",), _topical),
    Metric("US", ("US",), _uniqueness),
    Metric("CS", ("CS-r", "CS-p"), _completeness, needs_gold=True),
    Metric("Entailment", ("Entailment",), _entailment, is_done=_entailment_done),
    Metric("SS", ("S-r", "S-p", "S-f1"), _surprisal, phase=SS),
    Metric("FS", ("FS",), _factualness),
    Metric("GS", ("GS",), _granularity),
    Metric("Coherence", ("Coherence",), _coherence),
    Metric("MINE", ("MINE",), _mine, mine_only=True, is_done=_mine_done),
)

LLM_METRICS = ("FS", "GS", "Coherence", "MINE")


def metric_by_name(name: str) -> Metric:
    return next(m for m in METRICS if m.name == name)
