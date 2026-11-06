"""The evaluation run: datasets one after another; within a dataset the systems
concurrently (bounded by ``max_concurrent_systems``), first all metrics except surprisal,
then surprisal; LLM requests of all systems bounded by one semaphore.

Blocking work (metric computations, model loading, embedding prefetch, result writes)
runs in threads, so the event loop keeps serving the LLM requests of other systems.
"""

from __future__ import annotations

import asyncio
import gc
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from eval_config import DatasetConfig, EvalConfig
from evaluation import summary as status
from evaluation.concurrency import DEFAULT_WINDOW
from evaluation.gpu import select_cuda_device_with_most_free_memory
from evaluation.inputs import load_gold, load_system_inputs, prepare_embeddings, triple_stats
from evaluation.llm_judges import Judge
from evaluation.metrics import LLM_METRICS, MAIN, METRICS, PHASES, SS, Metric, MetricContext, ModelManager
from evaluation.results import ResultStore, to_serializable, write_json_atomic
from evaluation.summary import RunSummary, short_error
from source import completeness as completeness_module
from source.embeddings import (
    clear_embeddings_cache,
    load_element_embeddings,
    save_embeddings_atomic,
    set_embeddings_path,
)

STATS_KEYS = ("avg_num_triples", "avg_num_tokens_per_triple")


@dataclass
class RunOptions:
    max_concurrent_systems: int = 1
    max_llm_concurrency: int = 16
    compact_results: bool = True
    save_stats_separate: bool = False
    mine_max_articles: Optional[int] = None
    llm_window: int = DEFAULT_WINDOW


@dataclass
class RunState:
    """Objects shared by all datasets and systems of one run."""
    config: EvalConfig
    options: RunOptions
    summary: RunSummary
    judge: Judge
    models: ModelManager = field(default_factory=ModelManager)


def fetch_llm_model_info(endpoint: str) -> Dict[str, Any]:
    """Ask ``/models`` which model the endpoint serves (blocking)."""
    try:
        with urllib.request.urlopen(endpoint.rstrip("/") + "/models", timeout=10) as resp:
            models = json.loads(resp.read()).get("data", [])
        if models:
            return {"id": models[0].get("id"), "root": models[0].get("root"), "endpoint": endpoint}
        return {"endpoint": endpoint, "error": "no models returned"}
    except Exception as exc:
        return {"endpoint": endpoint, "error": str(exc)}


# --------------------------------------------------------------------------- one system

async def _run_metric(metric: Metric, ctx: MetricContext, state: RunState) -> None:
    config, ds, system = state.config, ctx.dataset_config, ctx.inputs.system

    def mark(st: str, error: Optional[str] = None) -> None:
        for label in metric.labels():
            state.summary.mark(ds.name, system, label, st, error)

    if not getattr(config.metrics, metric.name):
        return
    if metric.mine_only and not ds.mine:
        return
    if metric.needs_gold and ctx.inputs.gold is None:
        mark(status.SKIPPED, f"No ground truth for {metric.name}")
        return
    if not config.metrics.should_recompute(metric.name) and metric.done(ctx):
        mark(status.FINISHED_PREVIOUS)
        return
    print(f"[{system}] Computing {metric.name}...")
    try:
        out = await metric.compute(ctx)
        ctx.store.update(out.dataset, out.per_sentence)
        await asyncio.to_thread(ctx.store.save)
        mark(status.FINISHED_THIS)
    except Exception as exc:
        err = short_error(exc)
        print(f"[{system}] {metric.name} failed: {err}")
        mark(status.FAILED, err)


async def evaluate_system(system: str, dataset_config: DatasetConfig, gold: Optional[Dict[str, Any]],
                          phase: str, state: RunState, executor: ThreadPoolExecutor) -> None:
    """Compute the metrics of ``phase`` for one system on one dataset, skipping those
    already in its results file."""
    config, options, name = state.config, state.options, dataset_config.name
    print(f"[{system}] Processing {dataset_config.display_name} ({phase} phase)...")
    inputs = await asyncio.to_thread(load_system_inputs, system, dataset_config, gold)
    if inputs is None:
        print(f"[{system}] Warning: extraction file not found: "
              f"{dataset_config.get_extraction_path(system)}, skipping")
        if phase == MAIN:
            state.summary.add_system_error(name, system, "extraction file not found")
        return

    results_dir = config.get_dataset_results_dir(name)
    store = await asyncio.to_thread(ResultStore, results_dir / f"{system}.json", name, options.compact_results)

    if not store.has(STATS_KEYS, per_sentence=False):
        dataset_values, per_sentence = await asyncio.to_thread(triple_stats, inputs.predictions)
        store.update(dataset_values, per_sentence)
        if options.save_stats_separate:
            stats_path = config.get_dataset_stats_dir(name) / f"stats_{system}.json"
            await asyncio.to_thread(write_json_atomic, stats_path, to_serializable(store.data))
        await asyncio.to_thread(store.save)
        for key in STATS_KEYS:
            state.summary.mark(name, system, key, status.FINISHED_THIS)
    else:
        for key in STATS_KEYS:
            state.summary.mark(name, system, key, status.FINISHED_PREVIOUS)

    ctx = MetricContext(config, dataset_config, inputs, store, executor, state.judge, state.models,
                        options.mine_max_articles)
    for metric in METRICS:
        if metric.phase == phase:
            await _run_metric(metric, ctx, state)
    print(f"[{system}] Finished {dataset_config.display_name} ({phase} phase)")


# --------------------------------------------------------------------------- one dataset

def _prepare_dataset_embeddings(dataset_config: DatasetConfig, config: EvalConfig,
                                systems: List[str]) -> Optional[Path]:
    """Point the embedding cache at the dataset's file, fetch missing vectors and load
    the cache once (blocking; run in a thread). Loading it here, before the systems
    start, keeps concurrent US/CS computations from each loading the multi-GB file."""
    path = dataset_config.embeddings_path or config.embeddings_path
    if config.emb_api_endpoint and path and not Path(path).exists():
        save_embeddings_atomic(path, {})  # first run: start an empty cache
    set_embeddings_path(path)
    completeness_module.clear_gt_caches()
    print(f"Embeddings file: {path}")
    if config.metrics.US or config.metrics.CS:
        prepare_embeddings(dataset_config, config, systems, path)
        try:
            load_element_embeddings()
        except Exception as exc:
            print(f"Warning: could not load embeddings ({short_error(exc)}); US and CS will fail")
    return path


def _release_dataset_embeddings() -> None:
    set_embeddings_path(None)
    completeness_module.clear_gt_caches()
    clear_embeddings_cache()
    gc.collect()


async def run_dataset(dataset_config: DatasetConfig, systems: List[str], state: RunState) -> None:
    config, options = state.config, state.options
    print(f"\n{'=' * 60}\nDataset: {dataset_config.display_name}\nSystems: {systems}\n"
          f"Max concurrent systems: {options.max_concurrent_systems}, "
          f"max LLM concurrency: {options.max_llm_concurrency}\n{'=' * 60}")
    try:
        await asyncio.to_thread(_prepare_dataset_embeddings, dataset_config, config, systems)
        results_dir = config.get_dataset_results_dir(dataset_config.name)
        results_dir.mkdir(parents=True, exist_ok=True)
        if options.save_stats_separate:
            config.get_dataset_stats_dir(dataset_config.name).mkdir(parents=True, exist_ok=True)
        config.save_config(results_dir / "eval_config.toml")
        gold = await asyncio.to_thread(load_gold, dataset_config)

        system_slots = asyncio.Semaphore(max(1, options.max_concurrent_systems))

        async def guarded(system: str, phase: str) -> None:
            async with system_slots:
                try:
                    await evaluate_system(system, dataset_config, gold, phase, state, executor)
                except Exception as exc:
                    err = short_error(exc)
                    print(f"[{system}] system-level failure: {err}")
                    state.summary.add_system_error(dataset_config.name, system, err)

        with ThreadPoolExecutor(max_workers=max(4, 2 * options.max_concurrent_systems)) as executor:
            for phase in PHASES:
                if phase == SS and not config.metrics.SS:
                    continue
                try:
                    await asyncio.gather(*(guarded(s, phase) for s in systems))
                finally:
                    if phase == MAIN:
                        state.models.unload_cross_encoder()
                    else:
                        state.models.unload_surprisal(config.ss_model)
    finally:
        await asyncio.to_thread(_release_dataset_embeddings)


# --------------------------------------------------------------------------- whole run

async def run_evaluation(config: EvalConfig, options: RunOptions,
                         dataset_names: Optional[List[str]] = None,
                         systems: Optional[List[str]] = None,
                         config_path: str = "") -> RunSummary:
    """Evaluate the configured datasets and systems; returns the run summary, which is
    also written to ``<results_dir>/evaluation_run_summary.json`` after every dataset."""
    config.results_dir.mkdir(parents=True, exist_ok=True)
    summary = RunSummary(config_path)
    summary_path = config.results_dir / "evaluation_run_summary.json"

    if config.metrics.Entailment or config.metrics.SS:
        select_cuda_device_with_most_free_memory()  # before any model touches CUDA

    # Resolve the served model once, in a thread, instead of inside the first of many
    # concurrent requests where the blocking lookup would stall the event loop.
    llm_model = config.llm_model
    if config.llm_endpoint and any(getattr(config.metrics, m) for m in LLM_METRICS):
        info = await asyncio.to_thread(fetch_llm_model_info, config.llm_endpoint)
        summary.llm_model_info = info
        print(f"LLM endpoint model: {info.get('root') or info.get('id') or info.get('error', 'unknown')}")
        llm_model = llm_model or info.get("id")

    judge = Judge(config.llm_endpoint, llm_model, asyncio.Semaphore(max(1, options.max_llm_concurrency)),
                  temperature=config.llm_temperature, window=options.llm_window)
    state = RunState(config, options, summary, judge)

    for name in dataset_names or list(config.datasets):
        if name not in config.datasets:
            print(f"Warning: dataset '{name}' not in config, skipping")
            continue
        dataset_config = config.datasets[name]
        try:
            await run_dataset(dataset_config, systems or dataset_config.systems, state)
        finally:
            summary.write(summary_path)  # keep progress even if a dataset failed
    summary.write(summary_path)
    return summary
