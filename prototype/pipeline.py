"""Config, orchestration, and IO for the modular LLM OpenIE pipeline."""

from __future__ import annotations

import asyncio
import json
import shutil
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from . import stages

# Every stage the pipeline knows how to run.
KNOWN_STAGES = ["atomise", "coref", "extract_entities", "extract_triples", "verify"]

# Default order used when [stages] is given as boolean toggles rather than an
# explicit list. An explicit [stages].pipeline list overrides this entirely and
# may list a stage more than once.
DEFAULT_ORDER = ["atomise", "coref", "extract_entities", "extract_triples", "verify"]


@dataclass
class PipelineConfig:
    input_path: Path
    output_dir: Path
    endpoint: str | None
    model: str
    max_concurrency: int
    max_tokens: int
    prompts_dir: str
    pipeline: list[str]
    fewshot: dict[str, bool]
    fewshot_k: int
    use_entities: bool
    entities_instruction: str
    output_filename: str = "extractions.json"
    input_limit: int = 0
    chunk_size: int = 200
    temperature: float = 0.1
    seed: int | None = None
    config_path: Path | None = field(default=None)

    @classmethod
    def from_toml(cls, path: str | Path) -> "PipelineConfig":
        path = Path(path).resolve()
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        base = path.parent

        def resolve(p: str) -> Path:
            q = Path(p)
            return q if q.is_absolute() else (base / q).resolve()

        inp = data.get("input", {})
        out = data.get("output", {})
        llm = data.get("llm", {})
        prm = data.get("prompts", {})
        st = data.get("stages", {})
        fs = data.get("fewshot", {})
        et = data.get("extract_triples", {})
        rt = data.get("runtime", {})

        # Output is a directory; results and a copy of the config are written into it.
        if out.get("dir"):
            output_dir = resolve(out["dir"])
            output_filename = str(out.get("filename", "extractions.json"))
        elif out.get("path"):  # backward compatible: a file path -> dir + name
            p = resolve(out["path"])
            output_dir, output_filename = p.parent, p.name
        else:
            raise KeyError("config [output] needs a 'dir' (or legacy 'path')")

        # The pipeline is an ordered list of stage names; a stage may appear more
        # than once. An explicit [stages].pipeline list takes precedence; otherwise
        # fall back to the boolean toggles, run in DEFAULT_ORDER.
        if isinstance(st.get("pipeline"), list):
            pipeline = [str(s) for s in st["pipeline"]]
            unknown = [s for s in pipeline if s not in KNOWN_STAGES]
            if unknown:
                raise ValueError(
                    f"unknown stage(s) in [stages].pipeline: {unknown}; "
                    f"known stages are {KNOWN_STAGES}"
                )
        else:
            pipeline = [s for s in DEFAULT_ORDER if bool(st.get(s, False))]

        # [stages.repeats] sets how many times each stage runs in a row. Every
        # occurrence in the pipeline is expanded to that many consecutive runs
        # (default 1). Useful for iterative stages, e.g. verify.
        repeats = st.get("repeats", {})
        if not isinstance(repeats, dict):
            raise ValueError("config [stages.repeats] must be a table of stage = count")
        unknown_r = [s for s in repeats if s not in KNOWN_STAGES]
        if unknown_r:
            raise ValueError(
                f"unknown stage(s) in [stages.repeats]: {unknown_r}; "
                f"known stages are {KNOWN_STAGES}"
            )

        def repeat_count(stage: str) -> int:
            return max(1, int(repeats.get(stage, 1)))

        pipeline = [s for stage in pipeline for s in [stage] * repeat_count(stage)]

        return cls(
            input_path=resolve(inp["path"]),
            output_dir=output_dir,
            output_filename=output_filename,
            endpoint=(llm.get("endpoint") or None),
            model=(llm.get("model") or ""),
            max_concurrency=int(llm.get("max_concurrency", 8)),
            max_tokens=int(llm.get("max_tokens", 800)),
            prompts_dir=str(resolve(prm.get("dir", "prompts/llm_openie"))),
            pipeline=pipeline,
            fewshot={s: bool(fs.get(s, False)) for s in KNOWN_STAGES},
            fewshot_k=int(fs.get("k", 0) or 0),
            use_entities=bool(et.get("use_entities", False)),
            entities_instruction=str(
                et.get("entities_instruction", "Only consider relations between these entities:")
            ),
            input_limit=int(inp.get("limit", 0) or 0),
            chunk_size=int(rt.get("chunk_size", 200)),
            temperature=float(llm.get("temperature", 0.1)),
            seed=(int(llm["seed"]) if llm.get("seed") is not None else None),
            config_path=path,
        )


def load_inputs(cfg: PipelineConfig) -> list[str]:
    """Return the list of source texts. Accepts a JSON list or a JSON dict."""
    data = json.loads(cfg.input_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        texts = list(data.keys())
    elif isinstance(data, list):
        texts = [x if isinstance(x, str) else str(x) for x in data]
    else:
        raise ValueError(f"Unsupported input JSON type: {type(data).__name__}")
    if cfg.input_limit and cfg.input_limit > 0:
        texts = texts[: cfg.input_limit]
    return texts


async def process_unit(text: str, cfg: PipelineConfig, sem: asyncio.Semaphore) -> tuple[str, list[list]]:
    """Run the configured stages on one source text; return (text, triples).

    Stages execute in the order given by ``cfg.pipeline`` and may repeat. Shared
    state flows between them: ``working`` is the rewritten text (atomise, coref),
    ``entities`` feeds the triple stage, and ``triples`` is read and rewritten by
    extract_triples and verify.
    """
    working = text
    entities: list[str] | None = None
    triples: list[list] = []

    for stage in cfg.pipeline:
        if stage == "atomise":
            atoms = await stages.atomise(working, cfg, sem)
            working = " ".join(atoms)
        elif stage == "coref":
            working = await stages.coref(working, cfg, sem)
        elif stage == "extract_entities":
            entities = await stages.extract_entities(working, cfg, sem)
        elif stage == "extract_triples":
            passed_entities = entities if (cfg.use_entities and entities) else None
            triples = await stages.extract_triples(working, passed_entities, cfg, sem)
        elif stage == "verify":
            if triples:
                triples = await stages.verify(working, triples, cfg, sem)

    return text, triples


async def run_pipeline(cfg: PipelineConfig) -> dict[str, list[list]]:
    texts = load_inputs(cfg)
    fewshot_on = [s for s in dict.fromkeys(cfg.pipeline) if cfg.fewshot.get(s)]
    print(
        f"[prototype] {len(texts)} units | stages: {' -> '.join(cfg.pipeline) or 'none'} | "
        f"few-shot: {', '.join(fewshot_on) or 'none'} | "
        f"entities->triples: {cfg.use_entities} | concurrency: {cfg.max_concurrency}"
    )

    sem = asyncio.Semaphore(cfg.max_concurrency)
    results: dict[str, list[list]] = {}
    start = time.time()
    done = 0

    for i in range(0, len(texts), cfg.chunk_size):
        chunk = texts[i : i + cfg.chunk_size]
        for text, triples in await asyncio.gather(
            *(process_unit(t, cfg, sem) for t in chunk)
        ):
            results[text] = triples
        done += len(chunk)
        rate = done / max(time.time() - start, 1e-6)
        print(f"[prototype] {done}/{len(texts)} units ({rate:.1f}/s)")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    results_path = cfg.output_dir / cfg.output_filename
    results_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[prototype] wrote {len(results)} units to {results_path}")

    # Save the exact config that produced this run, for reproducibility.
    if cfg.config_path and cfg.config_path.exists():
        saved_config = cfg.output_dir / cfg.config_path.name
        if saved_config.resolve() != cfg.config_path.resolve():
            shutil.copyfile(cfg.config_path, saved_config)
            print(f"[prototype] saved config to {saved_config}")
    return results
