"""Configuration of the evaluation runner (evaluate.py): the dataset, metric and run
settings and their loading from and saving to TOML."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import tomllib  # Python 3.11+
except ImportError:  # pragma: no cover
    import tomli as tomllib

METRIC_NAMES = ("TS", "US", "FS", "GS", "CS", "SS", "Entailment", "Coherence", "MINE")

# Result keys that belong to a metric, for should_recompute().
_METRIC_ALIASES = {"CS-r": "CS", "CS-p": "CS", "S-r": "SS", "S-p": "SS", "S-f1": "SS", "SS_F1": "SS"}


@dataclass
class DatasetConfig:
    """Configuration for a single dataset."""
    name: str
    display_name: str
    ground_truth: Optional[Path]          # None for datasets without gold triples (MINE)
    extractions_dir: Path
    extraction_template: str
    lda_model: str
    systems: list[str]
    embeddings_path: Optional[Path] = None
    mine: bool = False  # MINE dataset: graph-form extractions, facts instead of gold triples
    # Optional alias -> exact extraction file. An alias listed here uses this file instead
    # of extractions_dir/extraction_template, so extraction files can live anywhere.
    system_paths: dict[str, Path] = field(default_factory=dict)

    def get_extraction_path(self, system: str) -> Path:
        """The extraction file of a system: its system_paths entry, else the template."""
        if system in self.system_paths:
            return self.system_paths[system]
        return self.extractions_dir / self.extraction_template.format(system=system)


@dataclass
class MetricsConfig:
    """Which metrics to compute (a metric missing from the TOML is on), and which to
    recompute although their results exist."""
    TS: bool = True          # topical score (LDA)
    US: bool = True          # uniqueness (embeddings)
    FS: bool = True          # factualness (LLM judge)
    GS: bool = True          # granularity (LLM judge)
    CS: bool = True          # completeness (embeddings, gold)
    SS: bool = True          # surprisal (causal LM)
    Entailment: bool = True  # NLI entailment (cross-encoder)
    Coherence: bool = True   # coherence (LLM judge)
    MINE: bool = True        # MINE knowledge retention (MINE dataset only)

    recompute_TS: bool = False
    recompute_US: bool = False
    recompute_FS: bool = False
    recompute_GS: bool = False
    recompute_CS: bool = False
    recompute_SS: bool = False
    recompute_Entailment: bool = False
    recompute_Coherence: bool = False
    recompute_MINE: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, bool]) -> "MetricsConfig":
        values: Dict[str, bool] = {}
        for name in METRIC_NAMES:
            values[name] = bool(data.get(name, True))
            # accept the older spelling <metric>_recompute as well
            values[f"recompute_{name}"] = bool(data.get(f"recompute_{name}", data.get(f"{name}_recompute", False)))
        return cls(**values)

    def to_dict(self) -> Dict[str, bool]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def should_recompute(self, metric_name: str) -> bool:
        """Whether a metric is to be recomputed; accepts result keys such as ``S-r``."""
        base = _METRIC_ALIASES.get(metric_name, metric_name)
        return bool(getattr(self, f"recompute_{base}", False))


@dataclass
class EvalConfig:
    """Full evaluation configuration loaded from TOML."""
    results_dir: Path
    stats_dir: Path
    topical_dir: Path
    llm_endpoint: str
    llm_model: Optional[str]
    ss_model: str
    embeddings_path: Path
    emb_api_endpoint: Optional[str] = None
    emb_api_model: str = "llm5"
    emb_api_save_new: bool = True
    max_concurrent_systems: int = 5
    max_llm_concurrency: int = 10
    entailment_batch_size: int = 128
    save_stats_separate: bool = True
    compact_results: bool = False
    # Sampling temperature sent to the LLM endpoint for all prompt-based metrics.
    llm_temperature: float = 0.3
    # Documentation only: which model version the endpoint served during the run.
    llm_endpoint_model_version: Optional[str] = None
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    datasets: Dict[str, DatasetConfig] = field(default_factory=dict)
    _source_path: Optional[Path] = field(default=None, repr=False)

    def get_dataset_results_dir(self, dataset_name: str) -> Path:
        return self.results_dir / dataset_name

    def get_dataset_stats_dir(self, dataset_name: str) -> Path:
        return self.stats_dir / dataset_name

    def to_dict(self) -> Dict[str, Any]:
        """The configuration as a TOML-ready dict; loading it with from_toml gives the
        same configuration (paths are written as absolute paths)."""
        general: Dict[str, Any] = {
            "results_dir": str(self.results_dir),
            "stats_dir": str(self.stats_dir),
            "topical_dir": str(self.topical_dir),
            "llm_endpoint": self.llm_endpoint,
            "ss_model": self.ss_model,
            "max_concurrent_systems": self.max_concurrent_systems,
            "max_llm_concurrency": self.max_llm_concurrency,
            "entailment_batch_size": self.entailment_batch_size,
            "save_stats_separate": self.save_stats_separate,
            "compact_results": self.compact_results,
            "llm_temperature": self.llm_temperature,
        }
        if self.llm_model:
            general["llm_model"] = self.llm_model
        if self.llm_endpoint_model_version:
            general["llm_endpoint_model_version"] = self.llm_endpoint_model_version
        embeddings: Dict[str, Any] = {
            "path": str(self.embeddings_path),
            "api_model": self.emb_api_model,
            "save_new": self.emb_api_save_new,
        }
        if self.emb_api_endpoint:
            embeddings["api_endpoint"] = self.emb_api_endpoint
        datasets = {}
        for name, ds in self.datasets.items():
            entry: Dict[str, Any] = {
                "display_name": ds.display_name,
                "extractions_dir": str(ds.extractions_dir),
                "extraction_template": ds.extraction_template,
                "lda_model": ds.lda_model,
                "systems": list(ds.systems),
                "mine": ds.mine,
            }
            if ds.ground_truth is not None:
                entry["ground_truth"] = str(ds.ground_truth)
            if ds.embeddings_path:
                entry["embeddings_path"] = str(ds.embeddings_path)
            if ds.system_paths:
                entry["system_paths"] = {k: str(v) for k, v in ds.system_paths.items()}
            datasets[name] = entry
        return {"general": general, "embeddings": embeddings,
                "metrics": self.metrics.to_dict(), "datasets": datasets}

    def save_config(self, output_path: Path) -> None:
        """Save the configuration to a TOML file (needs tomli-w)."""
        import tomli_w
        with open(output_path, "wb") as f:
            tomli_w.dump(self.to_dict(), f)

    @classmethod
    def from_toml(cls, toml_path: Path | str, base_dir: Optional[Path] = None) -> "EvalConfig":
        """Load the configuration; relative paths resolve against ``base_dir``
        (default: the working directory)."""
        toml_path = Path(toml_path)
        base_dir = Path.cwd() if base_dir is None else Path(base_dir)
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
        general = data.get("general", {})
        embeddings = data.get("embeddings", {})

        def resolve(p: str) -> Path:
            path = Path(p)
            return path if path.is_absolute() else base_dir / path

        config = cls(
            results_dir=resolve(general.get("results_dir", "./results")),
            stats_dir=resolve(general.get("stats_dir", "./stats")),
            topical_dir=resolve(general.get("topical_dir", "./topical")),
            llm_endpoint=general.get("llm_endpoint", "https://llm3-compute.cms.hu-berlin.de/v1/"),
            llm_model=general.get("llm_model") or None,
            ss_model=general.get("ss_model", "gpt2"),
            embeddings_path=resolve(embeddings.get("path", "./embeddings_dict_hu.json")),
            emb_api_endpoint=embeddings.get("api_endpoint") or None,
            emb_api_model=embeddings.get("api_model", "llm5"),
            emb_api_save_new=embeddings.get("save_new", True),
            max_concurrent_systems=int(general.get("max_concurrent_systems", 5)),
            max_llm_concurrency=int(general.get("max_llm_concurrency", 10)),
            entailment_batch_size=int(general.get("entailment_batch_size", 128)),
            save_stats_separate=general.get("save_stats_separate", True),
            compact_results=general.get("compact_results", False),
            llm_temperature=float(general.get("llm_temperature", 0.3)),
            llm_endpoint_model_version=general.get("llm_endpoint_model_version") or None,
            metrics=MetricsConfig.from_dict(data.get("metrics", {})),
            _source_path=toml_path,
        )

        for name, ds in data.get("datasets", {}).items():
            # ground_truth and lda_model are optional (MINE has neither): the metrics that
            # need them are then skipped. extractions_dir/extraction_template are optional
            # when system_paths names every system's file.
            gt = ds.get("ground_truth")
            emb = ds.get("embeddings_path")
            extractions_dir = ds.get("extractions_dir")
            system_paths = {alias: resolve(p) for alias, p in ds.get("system_paths", {}).items()}
            config.datasets[name] = DatasetConfig(
                name=name,
                display_name=ds.get("display_name", name),
                ground_truth=resolve(gt) if gt else None,
                extractions_dir=resolve(extractions_dir) if extractions_dir else base_dir,
                extraction_template=ds.get("extraction_template", "{system}.json"),
                lda_model=ds.get("lda_model", ""),
                systems=ds.get("systems", list(system_paths)),
                embeddings_path=resolve(emb) if emb else None,
                mine=bool(ds.get("mine", False)),
                system_paths=system_paths,
            )
        return config
