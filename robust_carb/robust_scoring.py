"""
CaRB and ROBUST scores on the ROBUST dataset (Qi et al., 2023). Works only on ROBUST:
it needs the paraphrase cliques of data/ROBUST/ROBUST.json, so the other datasets cannot
be scored with it. It is separate from evaluate.py.

For each system in data/ROBUST/ROBUST_<system>.json, computes per clique:
  - CaRB scores on the original sentence of the clique
  - Robust scores (the clique member with the worst CaRB F1)

and aggregates them as the official scorer of Qi et al. does: mean P and mean R over
cliques, F1 of those means.

Two operating points:
  - official (default): CaRB picks, for every sentence, the confidence threshold
    that maximises F1. Systems that emit varied confidences get a per-sentence
    oracle threshold; a system without confidences is scored on all its triples.
  - --keep-all: every extraction gets the same confidence, so the threshold sweep
    collapses to one point and every system is scored on all its triples. This is
    the setting the thesis reports.

Only a numeric (int/float) last element counts as a confidence. A string such as
"1886" is an argument and stays in the extraction.

Run from the cleaned_up/ directory:
    python robust_carb/robust_scoring.py                 # -> results/robust_carb/official/
    python robust_carb/robust_scoring.py --keep-all      # -> results/robust_carb/keep_all/
"""
import argparse
import json
import os
import sys
from operator import itemgetter

import numpy as np

# The bundled CaRB scorer imports its modules top-level (matcher, oie_readers.*)
CARB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CaRB")
sys.path.insert(0, CARB_DIR)

from carb import Benchmark
from oie_readers.extraction import Extraction
from matcher import Matcher

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "ROBUST")
GOLD_FILE = os.path.join(DATA_DIR, "ROBUST.json")
OUT_ROOT = os.path.join(PROJECT_ROOT, "results", "robust_carb")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def flat_to_extraction(triple, sent):
    """Convert a flat extraction triple to a CaRB Extraction object.

    Flat format (extraction files): [subject, predicate, obj, ..., confidence?]
    CaRB Extraction expects: pred first, then addArg for each argument.
    """
    # Only a numeric last element is a confidence score. A numeric *string*
    # ("1886", "42") is an argument: parsing it as a confidence would drop the
    # object and rank the triple by that number.
    if triple and isinstance(triple[-1], (int, float)) and not isinstance(triple[-1], bool):
        confidence = float(triple[-1])
        parts = triple[:-1]
    else:
        confidence = 1.0
        parts = triple

    if len(parts) < 2:
        return None

    subject = parts[0]
    predicate = parts[1]
    objects = parts[2:]

    ex = Extraction(pred=predicate, head_pred_index=-1, sent=sent, confidence=confidence)
    ex.addArg(subject)
    for obj in objects:
        ex.addArg(obj)
    return ex


def gold_to_extraction(args, sent):
    """Convert a gold args list to a CaRB Extraction object.

    Gold format (ROBUST.json): [predicate, arg1, arg2, ...]
    """
    if not args:
        return None
    predicate = args[0]
    ex = Extraction(pred=predicate, head_pred_index=-1, sent=sent, confidence=-1)
    for i in range(1, len(args)):
        ex.addArg(args[i])
    return ex


def score_sentence(pred_extractions, gold_extractions):
    """Run CaRB on a single sentence. Returns (auc, p, r, f1)."""
    bench = Benchmark()
    bench.gold = gold_extractions
    auc, (p, r, f1) = bench.compare(pred_extractions, Matcher.binary_linient_tuple_match)
    return float(auc), float(p), float(r), float(f1)


def build_gold_index(gold_data):
    """Build {sentence: {sent: [Extraction, ...]}} mapping for all cliques."""
    gold_index = {}  # ori_sent -> {sent -> [Extraction]}
    for clique in gold_data:
        ori_sent = clique["ori_sent"]
        clique_gold = {}

        # Original sentence
        ori_exts = [gold_to_extraction(a, ori_sent) for a in clique["ori_args"]]
        ori_exts = [e for e in ori_exts if e is not None]
        clique_gold[ori_sent] = {ori_sent: ori_exts}

        # Paraphrases
        for para in clique["paraphrases"]:
            sent = para["sent"]
            para_exts = [gold_to_extraction(a, sent) for a in para["args"]]
            para_exts = [e for e in para_exts if e is not None]
            clique_gold[sent] = {sent: para_exts}

        gold_index[ori_sent] = clique_gold
    return gold_index


def load_extractions(path):
    """Load flat extraction file: {sentence: [[subj, pred, obj, ..., conf], ...]}"""
    with open(path) as f:
        return json.load(f)


def build_pred_for_sentence(flat_extractions, sent):
    """Build CaRB-format prediction dict for a single sentence."""
    triples = flat_extractions.get(sent, [])
    exts = [flat_to_extraction(t, sent) for t in triples]
    exts = [e for e in exts if e is not None]
    return {sent: exts}


# ---------------------------------------------------------------------------
# Main scoring
# ---------------------------------------------------------------------------

def score_system(system_name, flat_extractions, gold_data, gold_index):
    """Score one system. Returns the full results dict ready to save."""
    per_sentence = {}
    all_carb_scores = []
    all_robust_scores = []

    for clique in gold_data:
        ori_sent = clique["ori_sent"]
        all_sents = [ori_sent] + [p["sent"] for p in clique["paraphrases"]]

        clique_scores = []  # (auc, p, r, f1) per member sentence

        for sent in all_sents:
            pred = build_pred_for_sentence(flat_extractions, sent)
            gold = gold_index[ori_sent][sent]
            auc, p, r, f1 = score_sentence(pred, gold)
            clique_scores.append((auc, p, r, f1))

        # CaRB = score on original sentence (first in list)
        carb_auc, carb_p, carb_r, carb_f1 = clique_scores[0]

        # Robust = member with worst F1
        worst = min(clique_scores, key=itemgetter(3))
        robust_auc, robust_p, robust_r, robust_f1 = worst

        # Build paraphrase detail list
        para_details = []
        for i, para in enumerate(clique["paraphrases"]):
            a, p_, r_, f1_ = clique_scores[i + 1]
            para_details.append({
                "sent": para["sent"],
                "auc": a, "p": p_, "r": r_, "f1": f1_,
            })

        per_sentence[ori_sent] = {
            "num_triples": len(flat_extractions.get(ori_sent, [])),
            "carb": {"auc": carb_auc, "p": carb_p, "r": carb_r, "f1": carb_f1},
            "robust": {"auc": robust_auc, "p": robust_p, "r": robust_r, "f1": robust_f1},
            "paraphrases": para_details,
        }

        all_carb_scores.append([carb_auc, carb_p, carb_r, carb_f1])
        all_robust_scores.append([robust_auc, robust_p, robust_r, robust_f1])

    carb_arr = np.array(all_carb_scores)
    robust_arr = np.array(all_robust_scores)

    def aggregate(arr):
        p = float(arr[:, 1].mean())
        r = float(arr[:, 2].mean())
        return {
            "auc": float(arr[:, 0].mean()),
            "p": p,
            "r": r,
            "f1": float(Benchmark.f1(p, r)),
        }

    return {
        system_name: {
            "per_sentence": per_sentence,
            "carb": aggregate(carb_arr),
            "robust": aggregate(robust_arr),
        }
    }


# ---------------------------------------------------------------------------
# Discover systems
# ---------------------------------------------------------------------------

def discover_systems(data_dir):
    """Return dict of system_name -> list of extraction file paths."""
    systems = {}

    for fname in sorted(os.listdir(data_dir)):
        if not fname.startswith("ROBUST_") or not fname.endswith(".json"):
            continue
        stem = fname[len("ROBUST_"):-len(".json")]

        # Skip non-extraction files
        if stem in ("processed", "all_text") or "embedding" in stem.lower():
            continue
        # Gold file
        if stem == "" or fname == "ROBUST.json":
            continue

        path = os.path.join(data_dir, fname)

        systems[stem] = [path]

    return systems


def load_and_merge(paths):
    """Load and merge flat extraction files."""
    merged = {}
    for path in paths:
        data = load_extractions(path)
        merged.update(data)
    return merged


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep-all", action="store_true",
                    help="score every system on all its extractions (constant confidence)")
    ap.add_argument("--out-dir", default=None, help="output folder (default results/robust_carb/<variant>)")
    ap.add_argument("--systems", nargs="*", default=None, help="subset of systems to score")
    args = ap.parse_args()

    global flat_to_extraction
    if args.keep_all:
        _with_confidence = flat_to_extraction

        def flat_to_extraction(triple, sent):  # noqa: F811
            ex = _with_confidence(triple, sent)
            if ex is not None:
                ex.confidence = 1.0
            return ex

    out_dir = args.out_dir or os.path.join(OUT_ROOT, "keep_all" if args.keep_all else "official")
    os.makedirs(out_dir, exist_ok=True)

    print("Loading gold data...")
    with open(GOLD_FILE) as f:
        gold_data = json.load(f)

    print("Building gold index...")
    gold_index = build_gold_index(gold_data)

    systems = discover_systems(DATA_DIR)
    if args.systems:
        systems = {k: v for k, v in systems.items() if k in args.systems}
    print(f"Found systems: {list(systems.keys())}\n")

    for system_name, paths in systems.items():
        print(f"Scoring {system_name}...")
        flat_extractions = load_and_merge(paths)

        results = score_system(system_name, flat_extractions, gold_data, gold_index)

        agg = results[system_name]
        print(f"  CaRB   -> P={agg['carb']['p']:.4f}  R={agg['carb']['r']:.4f}  F1={agg['carb']['f1']:.4f}")
        print(f"  Robust -> P={agg['robust']['p']:.4f}  R={agg['robust']['r']:.4f}  F1={agg['robust']['f1']:.4f}")

        out_path = os.path.join(out_dir, f"{system_name}.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"  Saved -> {out_path}\n")

    print("Done.")


if __name__ == "__main__":
    main()
