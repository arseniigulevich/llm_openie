"""Async stage functions. Each builds its prompt, calls the LLM, and parses.

A stage receives the pipeline ``cfg`` (any object exposing the attributes used
below) and a shared ``asyncio.Semaphore`` that bounds total LLM concurrency.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make ``source.hu_chat`` importable regardless of the current working directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from source.hu_chat import hu_instruct_async  # noqa: E402

from . import parsing  # noqa: E402
from .prompts import build_prompt  # noqa: E402


async def _ask(stage, fields, cfg, sem, *, entities=None):
    prompt = build_prompt(
        stage,
        fields,
        cfg.prompts_dir,
        use_fewshot=bool(cfg.fewshot.get(stage, False)),
        fewshot_k=cfg.fewshot_k,
        entities=entities,
        entities_instruction=cfg.entities_instruction,
    )
    return await hu_instruct_async(
        prompt,
        args_base_url=cfg.endpoint,
        model=cfg.model or None,
        semaphore=sem,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        seed=cfg.seed,
    )


async def atomise(text, cfg, sem) -> list[str]:
    raw = await _ask("atomise", {"$SENTENCE$": text}, cfg, sem)
    return parsing.parse_atomic(raw) or [text]


async def coref(text, cfg, sem) -> str:
    raw = await _ask("coref", {"$TEXT$": text}, cfg, sem)
    return raw.strip() or text


async def extract_entities(text, cfg, sem) -> list[str]:
    raw = await _ask("extract_entities", {"$TEXT$": text}, cfg, sem)
    return parsing.parse_entities(raw)


async def extract_triples(text, entities, cfg, sem) -> list[list]:
    raw = await _ask("extract_triples", {"$SENTENCES$": text}, cfg, sem, entities=entities)
    return parsing.parse_triples(raw)


async def verify(text, triples, cfg, sem) -> list[list]:
    if not triples:
        return []
    tuples_str = "\n".join(json.dumps(t, ensure_ascii=False) for t in triples)
    raw = await _ask("verify", {"$TEXT$": text, "$TUPLES$": tuples_str}, cfg, sem)
    supported, missing = parsing.parse_verify(raw)
    if not supported and not missing:
        return triples  # fall back to the input if the response was unusable
    return parsing._dedupe(supported + missing)
