#!/usr/bin/env python3
"""Convert prototype extractions into MINE graph-form for the MINE evaluation.

The prototype writes ``{essay_content: [[s, p, o], ...]}``. The MINE evaluation
(``source.mine_eval`` / the ``mine``-flagged dataset in evaluate.py) expects
``{article_id: {essay_topic, entities, edges, relations}}``. This maps each
prototype output essay back to its MINE article id (by exact essay_content match)
and builds the graph fields from the triples.

Usage (from the cleaned_up/ directory):
  python tools/prototype_to_mine_graph.py \
      --extractions prototype/results/mine_full/extractions.json \
      --out data/MINE/core_mine.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MINE_DATASET = REPO / "data" / "MINE" / "mine_dataset_full.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extractions", required=True, help="prototype extractions.json ({text: triples})")
    ap.add_argument("--out", required=True, help="output graph-form {id: {...}} for MINE eval")
    args = ap.parse_args()

    mine = json.loads(MINE_DATASET.read_text(encoding="utf-8"))
    by_text = {item["essay_content"]: item for item in mine}

    ext = json.loads(Path(args.extractions).read_text(encoding="utf-8"))

    out: dict[str, dict] = {}
    matched = 0
    for text, triples in ext.items():
        meta = by_text.get(text)
        if meta is None:
            continue
        matched += 1
        rels = [
            [str(t[0]).strip(), str(t[1]).strip(), str(t[2]).strip()]
            for t in triples
            if isinstance(t, (list, tuple)) and len(t) >= 3 and all(str(x).strip() for x in t[:3])
        ]
        entities = sorted({r[0] for r in rels} | {r[2] for r in rels})
        edges = sorted({r[1] for r in rels})
        out[str(meta["id"])] = {
            "essay_topic": meta.get("essay_topic", ""),
            "entities": entities,
            "edges": edges,
            "relations": rels,
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"matched {matched}/{len(ext)} essays -> {args.out} ({len(out)} articles)")
    if matched < len(ext):
        print("  (unmatched essays were skipped: essay_content did not match a MINE article)")


if __name__ == "__main__":
    main()
