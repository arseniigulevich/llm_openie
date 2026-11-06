"""Prompt loading, few-shot injection, and entity injection.

Every prompt under ``prompts/llm_openie`` is zero-shot by default and contains a
single ``Source text:`` anchor that marks the start of the live-input block, and
a single output-label line that the model continues from. Few-shot examples are
rendered from the sibling ``*.examples.json`` file and inserted immediately
before the ``Source text:`` anchor, so a prompt reads identically whether it runs
zero- or few-shot.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

# stage -> (prompt file, examples file, output label that the model continues from)
STAGE_SPECS: dict[str, dict[str, str]] = {
    "atomise": {
        "template": "atomize_sentence.txt",
        "examples": "atomize_sentence.examples.json",
        "label": "Atomic sentences:",
    },
    "coref": {
        "template": "coref_resolution.txt",
        "examples": "coref_resolution.examples.json",
        "label": "Resolved text:",
    },
    "extract_entities": {
        "template": "extract_entities.txt",
        "examples": "extract_entities.examples.json",
        "label": "Entities:",
    },
    "extract_triples": {
        "template": "extract_triples.txt",
        "examples": "extract_triples.examples.json",
        "label": "Triples:",
    },
    "verify": {
        "template": "verify_triples.txt",
        "examples": "verify_triples.examples.json",
        "label": "Verification:",
    },
}

ANCHOR = "Source text:"


@lru_cache(maxsize=None)
def load_template(stage: str, prompts_dir: str) -> str:
    path = Path(prompts_dir) / STAGE_SPECS[stage]["template"]
    return path.read_text(encoding="utf-8")


@lru_cache(maxsize=None)
def render_fewshot(stage: str, prompts_dir: str, k: int) -> str:
    """Render the few-shot example block for a stage (cached).

    Returns the concatenation of each example formatted with the file's
    ``render`` template. ``k <= 0`` uses every example.
    """
    path = Path(prompts_dir) / STAGE_SPECS[stage]["examples"]
    if not path.exists():
        return ""
    cfg = json.loads(path.read_text(encoding="utf-8"))
    render = cfg.get("render", "")
    examples = cfg.get("examples", [])
    if k and k > 0:
        examples = examples[:k]
    return "".join(render.format(**ex) for ex in examples)


def build_prompt(
    stage: str,
    fields: dict[str, str],
    prompts_dir: str,
    *,
    use_fewshot: bool = False,
    fewshot_k: int = 0,
    entities: list[str] | None = None,
    entities_instruction: str | None = None,
) -> str:
    """Assemble the final prompt for one stage.

    Order of operations: (1) inject few-shot examples before the ``Source text:``
    anchor, (2) inject the entity-constraint block before the live output label,
    (3) substitute the live ``$PLACEHOLDER$`` fields.
    """
    prompt = load_template(stage, prompts_dir)

    if use_fewshot:
        block = render_fewshot(stage, prompts_dir, fewshot_k)
        if block:
            prompt = prompt.replace(ANCHOR, block + ANCHOR, 1)

    if entities:
        label = STAGE_SPECS[stage]["label"]
        instruction = entities_instruction or "Only consider relations between these entities:"
        constraint = instruction + "\n" + "\n".join(entities) + "\n"
        # The live label is the LAST occurrence (few-shot examples also end on it).
        idx = prompt.rfind(label)
        if idx != -1:
            prompt = prompt[:idx] + constraint + prompt[idx:]

    for key, value in fields.items():
        prompt = prompt.replace(key, value)

    return prompt
