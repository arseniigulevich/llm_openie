"""Run the configured pipeline on one sentence, printing each stage's RAW LLM
response (before parsing).

Usage: python -m prototype.trace_run <config.toml> [sentence] [out_file]

The full trace is printed and also saved. By default it is written to
``trace.txt`` in the config's [output] dir; pass a third argument to override the
path.

Faithful trace: we wrap ``stages._ask`` so every LLM call prints its raw response,
then run the *real* stage functions. The printed raw is exactly the response that
gets parsed (one LLM call per stage, no double-calling).
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from . import stages
from .pipeline import PipelineConfig


class _Tee:
    """Mirror everything written to stdout into a file as well."""

    def __init__(self, stream, fh):
        self._stream = stream
        self._fh = fh

    def write(self, data):
        self._stream.write(data)
        self._fh.write(data)

    def flush(self):
        self._stream.flush()
        self._fh.flush()

SENT = (
    "In 1918 , the municipality was renamed Jones in honor of American "
    "congressman William Jones , who authored the Philippine Autonomy Act of "
    "1916 that provided for greater autonomy for the Philippines under American "
    "colonial rule ."
)

_orig_ask = stages._ask


async def _ask_traced(stage, fields, cfg, sem, *, entities=None):
    raw = await _orig_ask(stage, fields, cfg, sem, entities=entities)
    print("\n" + "=" * 80)
    print(f"STAGE: {stage}   (LLM call)")
    print("=" * 80)
    print("RAW RESPONSE:\n" + raw)
    return raw


async def run(cfg: PipelineConfig, text: str) -> None:
    sem = asyncio.Semaphore(cfg.max_concurrency)
    stages._ask = _ask_traced  # wrap so every call prints its raw response

    print(f"pipeline: {' -> '.join(cfg.pipeline)}")
    print(f"\nINPUT:\n{text}")

    working = text
    entities = None
    triples: list[list] = []

    for stage in cfg.pipeline:
        if stage == "atomise":
            atoms = await stages.atomise(working, cfg, sem)
            working = " ".join(atoms)
            print("\n-> working text:\n" + working)
        elif stage == "coref":
            working = await stages.coref(working, cfg, sem)
            print("\n-> working text:\n" + working)
        elif stage == "extract_entities":
            entities = await stages.extract_entities(working, cfg, sem)
            print("\n-> entities:\n" + json.dumps(entities, ensure_ascii=False))
        elif stage == "extract_triples":
            passed = entities if (cfg.use_entities and entities) else None
            triples = await stages.extract_triples(working, passed, cfg, sem)
            print("\n-> triples:\n" + json.dumps(triples, ensure_ascii=False, indent=2))
        elif stage == "verify":
            triples = await stages.verify(working, triples, cfg, sem)
            print("\n-> triples:\n" + json.dumps(triples, ensure_ascii=False, indent=2))

    print("\n" + "=" * 80)
    print("FINAL TRIPLES:")
    print(json.dumps(triples, ensure_ascii=False, indent=2))


def main() -> None:
    cfg = PipelineConfig.from_toml(sys.argv[1])
    text = sys.argv[2] if len(sys.argv) > 2 else SENT
    out_path = Path(sys.argv[3]) if len(sys.argv) > 3 else cfg.output_dir / "trace.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    orig_stdout = sys.stdout
    with out_path.open("w", encoding="utf-8") as fh:
        sys.stdout = _Tee(orig_stdout, fh)
        try:
            asyncio.run(run(cfg, text))
        finally:
            sys.stdout = orig_stdout
    print(f"\n[trace] saved to {out_path}")


if __name__ == "__main__":
    main()
