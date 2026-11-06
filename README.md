# data, prototype and evaluation

The repository contains the data of the six datasets, the reference extractions, the code of the LLM OpenIE prototype, and the evaluation code. 

## Layout

```
data/
  benchieFL/  lsoie/  RE-OIE-2016/  ROBUST/  WiRE57/  MINE/
prototype/            the five-stage LLM OpenIE pipeline (Python package)
  config/             run configurations (config.toml, config_mine.toml)
prompts/llm_openie/   stage prompt templates and few-shot examples used by the prototype
source/               LLM endpoint client (hu_chat.py) and the metric modules
evaluate.py           evaluation runner (see EVALUATION.md); implementation in evaluation/
tests/                unit and end-to-end tests (python -m pytest)
eval_config.py        TOML config loader of the runner
descriptive_subset.py derives the descriptive subset and its reference scores (Subset, CS F1, R_norm)
config/               eval_descriptive_subset.toml, the config for the descriptive subset
prompts/              judge prompts of FS, GS, Coherence and the MINE judge
topical/              per-dataset LDA models for TS and their training script
tools/                prototype_to_mine_graph.py (prototype output -> MINE graph form)
kg_gen/               minimal subset of KGGen used by the MINE score (MINE dataset only)
robust_carb/          CaRB/ROBUST scorer and robust_scoring.py (ROBUST dataset only)
requirements.txt      locked Python environment
```

The evaluation is documented in `EVALUATION.md`. Two scores are tied to one dataset: the
MINE score runs only on MINE (through `evaluate.py`, with `kg_gen/`), and the CaRB and ROBUST
scores run only on ROBUST (through `robust_carb/robust_scoring.py`).

The prototype expects `prompts/`, `data/` and `source/` next to `prototype/`: paths in its
config files are relative to `prototype/config/` (so `../../prompts/llm_openie` is
`cleaned_up/prompts/llm_openie`), and it imports the LLM client `source/hu_chat.py` from
`cleaned_up/`.

## Data files per dataset

| File pattern | Content |
|---|---|
| `*_processed.json` (WiRe57 documents: `doc_level_ground_truth.json`) | gold annotation |
| `*_all_text.json` | source texts, input to the systems |
| `<dataset>_<system>.json` / `<system>_mine.json` / `<system>_doc_level.json` | extractions |
| `ROBUST/ROBUST.json` | ROBUST release with the paraphrase cliques |
| `MINE/mine_dataset_full.json`, `MINE/mine_essays_full.json` | MINE essays with their 15 facts |
| `WiRE57/wire57_raw.md` | WiRe57 source documents |

Systems: `clausie`, `openie5`, `openie6`, `stanford_openie`, `llm` (the prototype), and
`kggen` on MINE.

## Datasets

| Folder | Dataset | Texts scored | Gold | Source |
|---|---|---|---|---|
| `benchieFL/` | BenchIE^FL | 300 sentences | 1,821 fact synsets | Lamarche and Langlais (2024), a re-annotation of BenchIE (Gashteovski et al., 2022) |
| `lsoie/` | LSOIE | 2,944 sentences | 8,479 triples | Solawetz and Larson (2021) |
| `RE-OIE-2016/` | Re-OIE2016 | 595 sentences | 1,488 triples | Zhan and Zhao (2020), a re-annotation of OIE2016 (Stanovsky and Dagan, 2016) |
| `ROBUST/` | ROBUST | 4,932 sentences in 1,272 cliques | 16,191 triples | Qi et al. (2023) |
| `WiRE57/` | WiRe57 | 5 documents (57 sentences) | 328 triples | Léchelle et al. (2019) |
| `MINE/` | MINE | 100 essays | 15 facts per essay | Mo et al. (2025) |

## Scores

| Score | Source |
|---|---|
| Topical similarity (TS), uniqueness (US), factualness (FS), granularity (GS), completeness recall (CS-r) | GenRES, Jiang et al. (2024); TS uses LDA (Blei et al., 2003) |
| Completeness precision (CS-p) | the precision of BenchIE, Gashteovski et al. (2022) |
| Entailment (Ent-s, Ent-c, Ent-t) | Pei et al. (2023), with the cross-encoder `nli-deberta-v3-large` (Reimers and Gurevych, 2019; He et al., 2023); premise blocks for long inputs follow Laban et al. (2022) |
| Coherence (Coh) | added in the thesis; the notion of entity-based coherence follows Jurafsky and Martin (2026) |
| Surprisal (S-r, S-p, S-f1) | added in the thesis; computed with Qwen2.5-1.5B (Qwen Team, 2024) |
| MINE | Mo et al. (2025) |
| CaRB | Bhardwaj et al. (2019) |
| ROBUST | Qi et al. (2023) |

The judged scores (FS, GS, Coherence, the MINE judge) used a fine-tuned Qwen3.6-27B (Qwen Team,
2026) on the Humboldt University endpoint; US and CS use Qwen3-Embedding-8B embeddings
(Zhang et al., 2025).

## Running the prototype

From this directory:

```bash
python -m prototype prototype/config/config.toml
```

Requires the `openai` Python package and access to the chat endpoint configured in
`source/hu_chat.py`. Results are written to `prototype/results/`.

## References

BibTeX entries as in the thesis bibliography (`citations.bib`), without the abstracts and the local file links.

```bibtex
@inproceedings{bhardwaj_carb_2019,
	address = {Hong Kong, China},
	title = {{CaRB}: {A} {Crowdsourced} {Benchmark} for {Open} {IE}},
	shorttitle = {{CaRB}},
	url = {https://aclanthology.org/D19-1651/},
	doi = {10.18653/v1/D19-1651},
	urldate = {2025-05-04},
	booktitle = {Proceedings of the 2019 {Conference} on {Empirical} {Methods} in {Natural} {Language} {Processing} and the 9th {International} {Joint} {Conference} on {Natural} {Language} {Processing} ({EMNLP}-{IJCNLP})},
	publisher = {Association for Computational Linguistics},
	author = {Bhardwaj, Sangnie and Aggarwal, Samarth and {Mausam}},
	editor = {Inui, Kentaro and Jiang, Jing and Ng, Vincent and Wan, Xiaojun},
	month = nov,
	year = {2019},
	pages = {6262--6267},
}

@inproceedings{gashteovski_benchie_2022,
	address = {Dublin, Ireland},
	title = {{BenchIE}: {A} {Framework} for {Multi}-{Faceted} {Fact}-{Based} {Open} {Information} {Extraction} {Evaluation}},
	shorttitle = {{BenchIE}},
	url = {https://aclanthology.org/2022.acl-long.307/},
	doi = {10.18653/v1/2022.acl-long.307},
	urldate = {2025-05-04},
	booktitle = {Proceedings of the 60th {Annual} {Meeting} of the {Association} for {Computational} {Linguistics} ({Volume} 1: {Long} {Papers})},
	publisher = {Association for Computational Linguistics},
	author = {Gashteovski, Kiril and Yu, Mingying and Kotnis, Bhushan and Lawrence, Carolin and Niepert, Mathias and Glavaš, Goran},
	editor = {Muresan, Smaranda and Nakov, Preslav and Villavicencio, Aline},
	month = may,
	year = {2022},
	pages = {4472--4490},
}

@inproceedings{jiang_genres_2024,
	address = {Mexico City, Mexico},
	title = {{GenRES}: {Rethinking} {Evaluation} for {Generative} {Relation} {Extraction} in the {Era} of {Large} {Language} {Models}},
	shorttitle = {{GenRES}},
	url = {https://aclanthology.org/2024.naacl-long.155/},
	doi = {10.18653/v1/2024.naacl-long.155},
	urldate = {2025-05-04},
	booktitle = {Proceedings of the 2024 {Conference} of the {North} {American} {Chapter} of the {Association} for {Computational} {Linguistics}: {Human} {Language} {Technologies} ({Volume} 1: {Long} {Papers})},
	publisher = {Association for Computational Linguistics},
	author = {Jiang, Pengcheng and Lin, Jiacheng and Wang, Zifeng and Sun, Jimeng and Han, Jiawei},
	editor = {Duh, Kevin and Gomez, Helena and Bethard, Steven},
	month = jun,
	year = {2024},
	pages = {2820--2837},
}

@article{laban_summac_2022,
	title = {{SummaC}: {Re}-{Visiting} {NLI}-based {Models} for {Inconsistency} {Detection} in {Summarization}},
	shorttitle = {{SummaC}},
	volume = {10},
	url = {https://aclanthology.org/2022.tacl-1.10/},
	doi = {10.1162/tacl_a_00453},
	journal = {Transactions of the Association for Computational Linguistics},
	author = {Laban, Philippe and Schnabel, Tobias and Bennett, Paul N. and Hearst, Marti A.},
	year = {2022},
	pages = {163--177},
}

@inproceedings{lamarche_benchiefl_2024,
	address = {Bangkok, Thailand},
	title = {{BenchIE$^{FL}$}: {A} {Manually} {Re}-{Annotated} {Fact}-{Based} {Open} {Information} {Extraction} {Benchmark}},
	shorttitle = {{BenchIE$^{FL}$}},
	url = {https://aclanthology.org/2024.findings-acl.496/},
	doi = {10.18653/v1/2024.findings-acl.496},
	urldate = {2025-05-30},
	booktitle = {Findings of the {Association} for {Computational} {Linguistics}: {ACL} 2024},
	publisher = {Association for Computational Linguistics},
	author = {Lamarche, Fabrice and Langlais, Philippe},
	editor = {Ku, Lun-Wei and Martins, Andre and Srikumar, Vivek},
	month = aug,
	year = {2024},
	pages = {8372--8394},
}

@inproceedings{lechelle2019wire57finegrainedbenchmark,
      title={{WiRe57}: A Fine-Grained Benchmark for Open Information Extraction},
      author={William L{\'e}chelle and Fabrizio Gotti and Philippe Langlais},
      booktitle={Proceedings of the 13th Linguistic Annotation Workshop},
      pages={6--15},
      year={2019},
      publisher={Association for Computational Linguistics},
      url={https://aclanthology.org/W19-4002/},
}

@inproceedings{mo_kggen_2025,
	title = {{KGGen}: {Extracting} {Knowledge} {Graphs} from {Plain} {Text} with {Language} {Models}},
	booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
	author = {Mo, Belinda and Yu, Kyssen and Kazdan, Joshua and Mpala, Proud and Yu, Lisa and Cundy, Chris and Kanatsoulis, Charilaos and Koyejo, Sanmi},
	year = {2025},
}

@inproceedings{pei-etal-2023-abstractive,
    title = "Abstractive Open Information Extraction",
    author = "Pei, Kevin  and
      Jindal, Ishan  and
      Chang, Kevin",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.376/",
    doi = "10.18653/v1/2023.emnlp-main.376",
    pages = "6146--6158",
}

@inproceedings{qi_robust_2023,
	address = {Singapore},
	title = {Preserving {Knowledge} {Invariance}: {Rethinking} {Robustness} {Evaluation} of {Open} {Information} {Extraction}},
	url = {https://aclanthology.org/2023.emnlp-main.360/},
	doi = {10.18653/v1/2023.emnlp-main.360},
	booktitle = {Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
	publisher = {Association for Computational Linguistics},
	author = {Qi, Ji and Zhang, Chuchun and Wang, Xiaozhi and Zeng, Kaisheng and Yu, Jifan and Liu, Jinxin and Hou, Lei and Li, Juanzi and Xu, Bin},
	month = dec,
	year = {2023},
	pages = {5876--5890},
}

@misc{qwen_qwen25_2024,
	title = {{Qwen2.5} {Technical} {Report}},
	url = {https://arxiv.org/abs/2412.15115},
	doi = {10.48550/arXiv.2412.15115},
	publisher = {arXiv},
	author = {{Qwen Team} and Yang, An and Yang, Baosong and Zhang, Beichen and Hui, Binyuan and Zheng, Bo and Yu, Bowen and Li, Chengyuan and Liu, Dayiheng and Huang, Fei and others},
	year = {2024},
	note = {arXiv:2412.15115},
}

@inproceedings{solawetz-larson-2021-lsoie,
    title = "{LSOIE}: A Large-Scale Dataset for Supervised Open Information Extraction",
    author = "Solawetz, Jacob  and
      Larson, Stefan",
    editor = "Merlo, Paola  and
      Tiedemann, Jorg  and
      Tsarfaty, Reut",
    booktitle = "Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume",
    month = apr,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.eacl-main.222/",
    doi = "10.18653/v1/2021.eacl-main.222",
    pages = "2595--2600",
}

@InProceedings{Stanovsky2016EMNLP,
  author    = {Gabriel Stanovsky and Ido Dagan},
  title     = {Creating a Large Benchmark for Open Information Extraction},
  booktitle = {Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  month     = {November},
  year      = {2016},
  address   = {Austin, Texas},
  publisher = {Association for Computational Linguistics},
  pages     = {2300--2305},
}

@inproceedings{zhan2019spanmodelopeninformation,
      title={Span Model for Open Information Extraction on Accurate Corpus},
      author={Junlang Zhan and Hai Zhao},
      booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
      volume={34},
      pages={9523--9530},
      year={2020},
      url={https://ojs.aaai.org/index.php/AAAI/article/view/6497},
}
```

## AI use disclosure

Co-Authored-By: Claude Opus 5.5 
