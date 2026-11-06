#!/usr/bin/env python3
"""Derive the descriptive subset score and its two reference scores from the results of
evaluate.py.

For every dataset and system in the config, reads results/<dataset>/<system>.json and
computes, from the per-text entries:

  Subset  mean over the non-empty extraction sets of mean(US, FS, Coherence, S-f1),
          the descriptive subset of the reference-free scores (Section 7.4.1)
  CS-f1   mean over the extraction sets of the harmonic mean of CS-p and CS-r
          (datasets with gold triples only)
  R_norm  mean over the essays of R / max(1, kappa), kappa = extracted tokens / essay
          tokens, the length-normalised MINE score (Equation eq:mine_norm); an empty
          extraction set counts as R = 0 (MINE only)

Run from the cleaned_up/ directory:
  python descriptive_subset.py --config config/eval_descriptive_subset.toml [--out results/descriptive_subset/descriptive_subset.json]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from eval_config import EvalConfig
from source.converter import _normalize_triple, _sanitize_field

SUBSET = ["US", "FS", "Coherence", "S-f1"]
TOKEN_RE = re.compile(r"\w+")


def _ntok(text: str) -> int:
    return len(TOKEN_RE.findall(text.lower()))


def _mean(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return float(np.mean(values)) if values else None


def subset_score(per_text: dict):
    out = []
    for entry in per_text.values():
        if not entry.get("num_triples"):
            continue
        vals = [entry.get(k) for k in SUBSET]
        if all(isinstance(v, (int, float)) for v in vals):
            out.append(float(np.mean(vals)))
    return _mean(out)


def cs_f1(per_text: dict):
    out = []
    for entry in per_text.values():
        p, r = entry.get("CS-p"), entry.get("CS-r")
        if isinstance(p, (int, float)) and isinstance(r, (int, float)):
            out.append(2 * p * r / (p + r) if p + r > 0 else 0.0)
    return _mean(out)


def mine_text_triples(graph_path: Path, mine_dir: Path) -> dict:
    """{sanitised essay text: normalised triples} from a graph-form MINE extraction file."""
    essays = {str(item["id"]): item["essay_content"]
              for item in json.loads((mine_dir / "mine_dataset_full.json").read_text(encoding="utf-8"))}
    graphs = json.loads(graph_path.read_text(encoding="utf-8"))
    out = {}
    for art_id, entry in graphs.items():
        text = essays.get(str(art_id))
        if text is None:
            continue
        triples = [_normalize_triple(r[:3]) for r in (entry.get("relations") or [])
                   if isinstance(r, (list, tuple)) and len(r) >= 3]
        out[_sanitize_field(text)] = [t for t in triples if t]
    return out


def r_norm(per_text: dict, triples_by_text: dict):
    out = []
    for text, entry in per_text.items():
        triples = triples_by_text.get(_sanitize_field(text), [])
        r = entry.get("MINE") if triples else 0.0
        if not isinstance(r, (int, float)):
            continue
        kappa = sum(_ntok(" ".join(t)) for t in triples) / max(1, _ntok(text))
        out.append(r / max(1.0, kappa))
    return _mean(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Derive the descriptive subset scores from evaluate.py results.")
    ap.add_argument("--config", required=True, help="the evaluation config used with evaluate.py")
    ap.add_argument("--out", default=None, help="JSON output (default: <results_dir>/descriptive_subset.json)")
    args = ap.parse_args()

    config = EvalConfig.from_toml(args.config)
    report = {}
    for name, ds in config.datasets.items():
        report[name] = {}
        for system in ds.systems:
            path = config.get_dataset_results_dir(name) / f"{system}.json"
            if not path.exists():
                print(f"[{name}] {system}: no results at {path}, skipped")
                continue
            results = json.loads(path.read_text(encoding="utf-8"))
            per_text = results.get(name, {}).get("per_sentence", {})
            scores = {"Subset": subset_score(per_text)}
            if ds.mine:
                scores["R_norm"] = r_norm(per_text, mine_text_triples(ds.get_extraction_path(system), ds.extractions_dir))
            else:
                scores["CS-f1"] = cs_f1(per_text)
            report[name][system] = scores

    out = Path(args.out) if args.out else config.results_dir / "descriptive_subset.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for name, systems in report.items():
        print(f"\n{config.datasets[name].display_name}")
        for system, scores in systems.items():
            cells = "  ".join(f"{k} {v:.3f}" if v is not None else f"{k} n/a" for k, v in scores.items())
            print(f"  {system:16s} {cells}")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
