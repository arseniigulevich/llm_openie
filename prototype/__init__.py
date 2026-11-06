"""Modular LLM Open Information Extraction pipeline.

Stages (each toggleable via the TOML config):
    atomise -> coref -> extract_entities -> extract_triples -> verify

The pipeline reads its prompts from ``prompts/llm_openie`` and talks to an
OpenAI-compatible endpoint through :mod:`source.hu_chat`. It is async and
bounded by a single concurrency semaphore so it stays fast.

Entry point: ``python -m prototype <config.toml>``.
"""

from .pipeline import PipelineConfig, run_pipeline

__all__ = ["PipelineConfig", "run_pipeline"]
