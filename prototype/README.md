# Modular LLM OpenIE Pipeline

A configurable, async Open Information Extraction pipeline driven by the prompts in
`prompts/llm_openie`. Every stage is toggleable, few-shot examples can be switched
on per stage at run time, and the extract-triples stage can be constrained to the
entities found earlier.

## Run

```bash
# from the repository root, in the environment that has `openai` installed
python -m prototype prototype/config/config.toml
```

The config file is the only argument. Relative paths inside it are resolved
against the config file's own directory.

## Stages

| Stage | Prompt | Effect |
|-------|--------|--------|
| `atomise` | `atomize_sentence.txt` | split the text into atomic sentences |
| `coref` | `coref_resolution.txt` | replace pronouns / descriptions with their referents |
| `extract_entities` | `extract_entities.txt` | list the entities (fed to the triple stage) |
| `extract_triples` | `extract_triples.txt` | produce `[subject, predicate, object]` triples |
| `verify` | `verify_triples.txt` | drop unsupported triples, add missing ones |

The text flows through the stages in order; later stages see the rewritten text
and triples from the earlier ones.

### Stage order

`[stages]` can be configured two ways:

1. **Explicit list** — `pipeline` is an ordered list of stage names. A stage may
   appear more than once, so a stage can be run repeatedly:

   ```toml
   [stages]
   pipeline = ["atomise", "coref", "extract_triples", "verify"]
   ```

2. **Boolean toggles** (back-compatible) — set each stage `true`/`false`. The
   enabled stages run in the default order:

   ```
   atomise -> coref -> extract_entities -> extract_triples -> verify
   ```

The explicit `pipeline` list overrides the boolean toggles when both are present.

### Repeating a stage

`[stages.repeats]` sets how many times each stage runs in a row (default 1).
Every occurrence of the stage in the resolved pipeline is expanded into that many
consecutive runs:

```toml
[stages.repeats]
verify = 2        # verify twice in a row to refine the triples
```

This works with both the explicit list and the boolean toggles. For example,
toggles `atomise, extract_triples, verify` with `verify = 2` run as
`atomise -> extract_triples -> verify -> verify`. Each run sees the output of the
previous one, so repeats are meant for iterative stages (`verify`, `coref`).
Repeating `extract_triples` re-extracts and overwrites.

## Few-shot examples

`[fewshot]` enables few-shot prompting per stage. When on for a stage, the
examples from its `*.examples.json` file are rendered and inserted immediately
before the `Source text:` line, so the prompt is identical in shape to the
zero-shot version. `k` caps the number of shots (`0` = all).

## Entities in the triple stage

With `[extract_triples] use_entities = true`, the entities produced by
`extract_entities` are appended to the triple prompt, right before the live
`Triples:` label, introduced by `entities_instruction`
(default: *"Only consider relations between these entities:"*). This realises an
entity-conditioned extraction similar to KGGen's entities-then-relations design.
It has no effect unless the `extract_entities` stage is also enabled.

## Output

`[output] dir` is a directory. The pipeline writes two files into it:

- `filename` (default `extractions.json`): the extractions, mapping each source
  text to its triples:

  ```json
  { "<source text>": [["subject", "predicate", "object"], ...] }
  ```

  This is the format consumed by the evaluation pipeline, so it can be renamed to
  `{dataset}_{system}.json` and scored directly.
- a copy of the config file that produced the run, for reproducibility.

A legacy `[output] path = "...json"` is still accepted and is split into the
directory and the filename.

## Performance

All LLM calls go through `source.hu_chat.hu_instruct_async` and share one
`asyncio.Semaphore(max_concurrency)`. Units within a chunk run concurrently;
stages within a unit run in sequence (they depend on each other). Tune
`[llm] max_concurrency` and `[runtime] chunk_size` for throughput.

## Module layout

| File | Responsibility |
|------|----------------|
| `prompts.py` | load prompts, render/inject few-shot, inject entity constraint |
| `parsing.py` | parse each stage's raw output (atomic sentences, entities, triples, verification) |
| `stages.py` | one async function per stage |
| `pipeline.py` | `PipelineConfig.from_toml`, IO, orchestration |
| `__main__.py` | CLI entry point |
