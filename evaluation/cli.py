"""Command line of the evaluation runner (``python evaluate.py --help``)."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import List, Optional

from eval_config import EvalConfig
from evaluation.runner import RunOptions, run_evaluation


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate OpenIE extractions (see EVALUATION.md).")
    p.add_argument("--config", default="config/eval_descriptive_subset.toml", help="TOML config file")
    p.add_argument("--datasets", nargs="+", default=None, help="datasets to evaluate (default: all)")
    p.add_argument("--systems", nargs="+", default=None, help="systems to evaluate (default: per dataset)")
    p.add_argument("--max-concurrent-systems", type=int, default=None,
                   help="systems evaluated concurrently within a dataset (overrides the config)")
    p.add_argument("--max-llm-concurrency", type=int, default=None,
                   help="concurrent requests to the chat endpoint (overrides the config)")
    p.add_argument("--entailment-batch-size", type=int, default=None,
                   help="cross-encoder batch size (overrides the config)")
    p.add_argument("--compact-results", dest="compact_results", action="store_true", default=None,
                   help="drop *_details keys from the result files")
    p.add_argument("--no-compact-results", dest="compact_results", action="store_false",
                   help="keep the full details in the result files")
    p.add_argument("--save-separate-stats", dest="save_separate_stats", action="store_true", default=None,
                   help="also write stats_dir/<dataset>/stats_<system>.json")
    p.add_argument("--no-separate-stats", dest="save_separate_stats", action="store_false",
                   help="do not write separate stats files")
    p.add_argument("--mine-max-articles", type=int, default=None,
                   help="score MINE on the first N articles only (for testing); "
                        "a later full run recomputes it")
    return p


def options_from_args(args: argparse.Namespace, config: EvalConfig) -> RunOptions:
    pick = lambda cli, cfg: cfg if cli is None else cli
    mine_max = args.mine_max_articles
    if mine_max is None and os.environ.get("MINE_MAX_ARTICLES"):  # older interface
        mine_max = int(os.environ["MINE_MAX_ARTICLES"])
    return RunOptions(
        max_concurrent_systems=pick(args.max_concurrent_systems, config.max_concurrent_systems),
        max_llm_concurrency=pick(args.max_llm_concurrency, config.max_llm_concurrency),
        compact_results=pick(args.compact_results, config.compact_results),
        save_stats_separate=pick(args.save_separate_stats, config.save_stats_separate),
        mine_max_articles=mine_max,
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        return 2
    config = EvalConfig.from_toml(config_path)
    batch = config.entailment_batch_size if args.entailment_batch_size is None else args.entailment_batch_size
    os.environ["OPENIE_ENTAILMENT_BATCH_SIZE"] = str(batch)  # read by source/entailment_eval.py
    options = options_from_args(args, config)
    summary = asyncio.run(run_evaluation(config, options, args.datasets, args.systems, str(config_path)))
    failed = sum(len(s["failed"]) + len(s["system_errors"])
                 for ds in summary.to_dict()["datasets"].values() for s in ds["systems"].values())
    print(f"\n{'=' * 60}\nEvaluation finished; {failed} failed metric(s), "
          f"see {config.results_dir / 'evaluation_run_summary.json'}\n{'=' * 60}")
    return 0
