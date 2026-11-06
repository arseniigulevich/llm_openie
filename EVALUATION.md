# Evaluation

`evaluate.py` scores the extractions of OpenIE systems with a subset of metrics. A TOML config names the datasets, the systems, the metrics to compute and the
model endpoints. `descriptive_subset.py` derives from the results the descriptive subset score and the two reference scores it is compared with, CS F1 and R_norm.
`config/eval_descriptive_subset.toml` computes these on BenchIE-FL, the WiRe57 documents and
MINE.

## Quick start: the descriptive subset


```bash
python evaluate.py --config config/eval_descriptive_subset.toml          # per-text scores
python descriptive_subset.py --config config/eval_descriptive_subset.toml   # Subset, CS F1, R_norm
```

`evaluate.py` writes `results/descriptive_subset/<dataset>/<system>.json` and
`results/descriptive_subset/evaluation_run_summary.json`. `descriptive_subset.py` prints a table and writes
`results/descriptive_subset/descriptive_subset.json`:

| Score | Definition | Datasets |
|---|---|---|
| Subset | mean over the non-empty extraction sets of mean(US, FS, Coherence, S F1) | all three |
| CS F1 | mean over the extraction sets of the harmonic mean of CS-p and CS-r | BenchIE-FL, WiRe57 |
| R_norm | mean over the essays of R / max(1, κ), κ = extracted tokens / essay tokens; an empty extraction set counts as R = 0 | MINE |

As shipped, the config scores the `llm` prototype.

```bash
python tools/prototype_to_mine_graph.py \
    --extractions prototype/results/mine_full/extractions.json --out data/MINE/core_mine.json
```

## Metrics

| Config key | Scores in the results | Needs | Module |
|---|---|---|---|
| `TS` | TS | LDA model in `topical/` | `source/topical.py` |
| `US` | US | embedding endpoint | `source/uniqueness.py` |
| `FS` | FS | chat endpoint, `prompts/fact_checker.txt` | `evaluation/llm_judges.py` |
| `GS` | GS | chat endpoint, `prompts/granularity_checker.txt` | `evaluation/llm_judges.py` |
| `Coherence` | Coherence | chat endpoint, `prompts/coherence_checker.txt` | `evaluation/llm_judges.py` |
| `CS` | CS-r, CS-p | gold triples, embedding endpoint | `source/completeness.py` |
| `SS` | S-r, S-p, S-f1 (atomicity-weighted; the raw values as `*_raw`) | causal LM (`ss_model`), GPU recommended | `source/surprisal.py` |
| `Entailment` | sentence, combined and tuple granularity (Ent-s, Ent-c, Ent-t) | cross-encoder `nli-deberta-v3-large`, GPU recommended; only Ent-s without gold | `source/entailment_eval.py` |
| `MINE` | MINE | MINE dataset only (`mine = true`), chat endpoint, bundled `kg_gen/` | `source/mine_eval.py` |

Metrics that need gold triples are skipped on datasets without a `ground_truth`.

**Dataset-specific scores.** Two scores apply to one dataset only:

- **MINE** works only on the MINE dataset. It needs the 15 facts per essay in
  `data/MINE/mine_dataset_full.json` and graph-form extractions, so `evaluate.py` computes it
  only for a dataset block with `mine = true`. Graph deduplication and retrieval use the
  bundled subset of KGGen in `kg_gen/` (see below).
- **CaRB and ROBUST** work only on the ROBUST dataset. They need the paraphrase cliques of
  `data/ROBUST/ROBUST.json` and are computed by `robust_carb/robust_scoring.py`, not by
  `evaluate.py` (see "CaRB and ROBUST scores").

## CaRB and ROBUST scores (ROBUST dataset only)

`robust_carb/robust_scoring.py` scores the systems in `data/ROBUST/ROBUST_<system>.json`
against the ROBUST gold with the CaRB matcher (`binary_linient_tuple_match`). Per clique it
reports the CaRB score on the original sentence and the ROBUST score, the CaRB score of the
clique member with the lowest F1; both are aggregated as mean P and mean R over the 1,272
cliques and the F1 of those means. It takes no config file.

```bash
python robust_carb/robust_scoring.py --keep-all    # -> results/robust_carb/keep_all/<system>.json
python robust_carb/robust_scoring.py               # -> results/robust_carb/official/<system>.json
```

`--keep-all` scores every extraction at the same confidence, so each system is scored on all
its triples; this is the setting the thesis reports. Without it, CaRB picks the
F1-maximising confidence threshold per sentence. `--systems` restricts the run, `--out-dir`
changes the output folder. 

`robust_carb/CaRB/` holds the CaRB scorer of Bhardwaj et al. (2019) as released with ROBUST
by Qi et al. (2023), reduced to `Benchmark`, `Matcher`, `Extraction`. Licences: MIT,
`robust_carb/CaRB/LICENSE` (CaRB) and `robust_carb/LICENSE_ROBUST` (ROBUST).

## Config file

```toml
[general]
results_dir = "./results/run"        # results/<dataset>/<system>.json
stats_dir   = "./results/run/stats"
topical_dir = "./topical"            # <lda_model>_lda.pkl and <lda_model>_dictionary.pkl
llm_endpoint = "https://.../v1/"     # OpenAI-compatible chat endpoint for FS, GS, Coherence, MINE
llm_model = ""                       # optional; empty = use the endpoint's default model
llm_temperature = 0.1
ss_model = "Qwen/Qwen2.5-1.5B"       # Hugging Face id of the surprisal LM
max_concurrent_systems = 1           # systems evaluated in parallel (keep 1 with MINE)
max_llm_concurrency = 16             # parallel requests to the chat endpoint
entailment_batch_size = 128
save_stats_separate = false          # also write stats/stats_<system>.json
compact_results = true               # drop bulky per-pair detail from the results

[embeddings]                         # element embeddings for US and CS
path = "./data/<dataset>/embeddings_dict_hu.json"   # default cache file
api_endpoint = "https://.../v1"      # embedding endpoint for missing vectors
api_model = "llm5"
save_new = true                      # write fetched vectors back to the cache

[metrics]                            # switch each metric on or off
TS = false
US = true
# ...
recompute_US = false                 # recompute even if a result already exists

[datasets.<name>]                    # one block per dataset
display_name = "..."
ground_truth = "./data/.../..._processed.json"   # omit for datasets without gold (MINE)
extractions_dir = "./data/..."
extraction_template = "<prefix>_{system}.json"   # file name of a system's extractions
systems = ["clausie", "llm"]
lda_model = "benchie"                # LDA model used by TS
embeddings_path = "./data/.../embeddings_dict_hu.json"
mine = false                         # true only for the MINE dataset (graph-form extractions)

[datasets.<name>.system_paths]       # optional: explicit file per system, overrides the template
my_system = "./path/to/extractions.json"
```

Results are written incrementally (and atomically) after every metric. A rerun skips a
metric whose dataset-level score and per-text entries are all in the results file, unless
`recompute_<metric>` is set, so an interrupted run can be resumed.
`results_dir/evaluation_run_summary.json` lists for every dataset and system which metrics
finished in this run, which came from an earlier run, which were skipped (e.g. CS on a
dataset without gold triples) and which failed, with the error.

Command-line options (`python evaluate.py --help`): `--datasets` and `--systems` restrict a
run to some of the configured datasets and systems; `--max-concurrent-systems`,
`--max-llm-concurrency` and `--entailment-batch-size` override the config;
`--mine-max-articles N` scores MINE on the first N articles only, for a quick test (a later
run with more articles recomputes it).

## Structure of the runner

`evaluate.py` is the entry point; the runner is the `evaluation/` package:

| Module | Role |
|---|---|
| `cli.py` | command line, CLI options over config values |
| `runner.py` | datasets in sequence; per dataset the systems concurrently, first all metrics except SS, then SS; skipping, saving and run-summary status per metric |
| `metrics.py` | registry of the metrics: config switch, result keys, phase, compute function |
| `llm_judges.py` | FS, GS, Coherence: prompts, response parsers, scores |
| `inputs.py` | loading extractions, gold and MINE graphs; triple statistics; embedding prefetch |
| `results.py` | result files: incremental atomic saving, compact form |
| `summary.py` | run summary |
| `concurrency.py` | bounded gathering of the LLM requests, cancelling the rest on a failure |
| `gpu.py` | GPU selection and memory release |

Concurrency: the requests of all systems to the chat endpoint share one semaphore
(`max_llm_concurrency`). Blocking work (metric computations, model loading, embedding
prefetch, result writes) runs in threads, so the event loop keeps serving the requests of
other systems. The entailment cross-encoder is loaded on first use and unloaded before the
surprisal phase, so the two models do not share the GPU; KGGen's deduplication for MINE runs
on one dedicated thread because DSPy settings are bound to the thread that set them.

## Tests

```bash
python -m pytest              # unit and end-to-end tests, about 30 s, no network or GPU
python -m pytest -m slow      # also MINE with KGGen, Entailment and surprisal (loads models)
```

The end-to-end tests run `evaluate.py` and `descriptive_subset.py` on a toy dataset against
`tests/llm_stub.py`, a local stand-in for the chat endpoint that answers every judge prompt
deterministically and counts requests (used to check that a resumed run sends none).

## Input formats

- Extractions: `{source text: [[subject, relation, object, ...], ...]}`. A numeric last element
  is a confidence and is dropped; all string elements are kept, so a triple may have two
  arguments (no object) or several objects. A number written as a string is an argument. On
  MINE: `{essay id: {essay_topic, entities, edges, relations}}`, as produced by
  `tools/prototype_to_mine_graph.py`, where each relation is `[subject, relation, object]`;
  extra elements are ignored.
- Gold: `{source text: [[s, r, o], ...]}`; for BenchIE-FL a list of fact synsets per text,
  each a list of equivalent triples.

## Requirements and reproducibility

`requirements.txt` lists the locked environment of the thesis (Python 3.13, CUDA). MINE uses
the bundled `kg_gen/` package, a subset of KGGen v0.4.0 (https://github.com/stair-lab/kg-gen,
commit 6259b4c, MIT licence in `kg_gen/LICENSE`) with only the graph model, deduplication,
embedding and retrieval code that `source/mine_eval.py` calls; generation, clustering,
visualisation and the Neo4j integration are removed. 

