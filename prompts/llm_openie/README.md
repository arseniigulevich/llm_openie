# LLM OpenIE Pipeline Prompts

Prompts for the staged LLM Open Information Extraction pipeline. The stages run in
this order:

| Stage | Prompt | Examples | Placeholders | Output label |
|-------|--------|----------|--------------|--------------|
| atomise | `atomize_sentence.txt` | `atomize_sentence.examples.json` | `$SENTENCE$` | `Atomic sentences:` |
| resolve coreferences | `coref_resolution.txt` | `coref_resolution.examples.json` | `$TEXT$` | `Resolved text:` |
| extract entities | `extract_entities.txt` | `extract_entities.examples.json` | `$TEXT$` | `Entities:` |
| extract triples | `extract_triples.txt` | `extract_triples.examples.json` | `$SENTENCES$`, `$ENTITIES$` (optional) | `Triples:` |
| entity/edge resolution | `entity_or_edge_resolution.txt` | `entity_or_edge_resolution.examples.json` | `$CANDIDATES$`, `$ITEM$`, `$ITEM_TYPE$` | `Duplicates:` |
| verify | `verify_triples.txt` | `verify_triples.examples.json` | `$TEXT$`, `$TUPLES$` | `Verification:` |

## Uniform prompt structure

Every `.txt` prompt follows the same layout and is zero-shot by default:

```
<persona>            # one line: who the model is acting as
<task + rules>       # what to do and the constraints
<output format>      # exactly what to emit
                     # <-- few-shot examples are injected here, before "Source text:"
Source text:
<placeholder(s)>     # the live input, plus any secondary labelled fields
<output label>:      # the prompt ends on this label; the model continues from here
```

The line `Source text:` is the **injection anchor**. It appears exactly once in
each prompt, as the start of the live-input block.

## Few-shot example files

Each `*.examples.json` is a sibling of its prompt and has the shape:

```json
{
  "stage": "<stage name>",
  "render": "Source text:\n{field}\n<output label>:\n{output}\n\n",
  "examples": [ { "field": "...", "output": "..." } ]
}
```

- `render` is a Python `str.format` template that reproduces the prompt's input
  block using the same labels, with one `{placeholder}` per field of the
  example and a final `{output}`.
- Each entry in `examples` supplies every field named in `render`.

## Injecting examples at run time

To turn a zero-shot prompt into a few-shot one, render the examples and insert the
block immediately before the `Source text:` anchor:

```python
import json

def build_prompt(prompt_path, examples_path=None, k=None):
    prompt = open(prompt_path, encoding="utf-8").read()
    if not examples_path:
        return prompt                      # zero-shot
    cfg = json.load(open(examples_path, encoding="utf-8"))
    shots = cfg["examples"][:k] if k else cfg["examples"]
    block = "".join(cfg["render"].format(**ex) for ex in shots)
    return prompt.replace("Source text:", block + "Source text:", 1)
```

Then substitute the live placeholders (`$SENTENCE$`, `$TEXT$`, …) as usual.
Because the rendered examples use the same `Source text:` / output-label format as
the live query, the prompt stays consistent whether it is zero- or few-shot.

The `extract_triples` stage takes optional entities: when running without them,
drop the `Known entities (optional):` line and the `$ENTITIES$` placeholder (and
use an examples file whose `render` omits the `{entities}` field).
