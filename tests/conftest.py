import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.llm_stub import StubLLMServer  # noqa: E402


@pytest.fixture(scope="session")
def llm_server():
    with StubLLMServer() as server:
        yield server


@pytest.fixture(autouse=True)
def _fresh_llm_clients():
    """hu_chat caches one AsyncOpenAI client per endpoint; its connection pool is bound
    to the event loop of the first run, so every test starts with a fresh cache."""
    from source import hu_chat
    hu_chat._async_client_cache.clear()
    hu_chat._model_cache.clear()
    yield
    hu_chat._async_client_cache.clear()


def vector(text: str, dim: int = 32) -> list:
    """Deterministic pseudo-random embedding of a string."""
    seed = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
    return np.random.default_rng(seed).normal(size=dim).tolist()


# A small dataset: three texts (one with a newline, which sanitisation turns into a
# space), gold triples, and two systems. "alpha" repeats one triple (once with a
# confidence) and has no triples for the third text.
TEXT_1 = "Marie Curie won the Nobel Prize in 1903.\nShe was born in Warsaw."
TEXT_2 = "The river flows into the sea."
TEXT_3 = "Nothing to see here."
GOLD = {
    TEXT_1: [["Marie Curie", "won", "the Nobel Prize"], ["Marie Curie", "was born in", "Warsaw"]],
    TEXT_2: [["The river", "flows into", "the sea"]],
    TEXT_3: [],
}
SYSTEMS = {
    "alpha": {
        TEXT_1: [["Marie Curie", "won", "the Nobel Prize", 0.9], ["Marie Curie", "won", "the Nobel Prize"],
                 ["Marie Curie", "was born in", "Warsaw"]],
        TEXT_2: [["The river", "flows into", "the sea"]],
        TEXT_3: [],
    },
    "beta": {
        TEXT_1: [["Curie", "won", "Nobel Prize in 1903"]],
        TEXT_2: [["river", "flows", "sea"], ["The river", "flows into", "the sea"]],
        TEXT_3: [["Nothing", "to see", "here"]],
    },
}


def sanitize(text: str) -> str:
    return " ".join(text.replace("\t", " ").replace("\n", " ").split())


@pytest.fixture
def toy_dataset(tmp_path):
    """Write the toy dataset with an embedding cache that covers every element."""
    data = tmp_path / "data"
    data.mkdir()
    (data / "toy_processed.json").write_text(json.dumps(GOLD), encoding="utf-8")
    for system, extractions in SYSTEMS.items():
        (data / f"toy_{system}.json").write_text(json.dumps(extractions), encoding="utf-8")
    elements = {e for triples in [*GOLD.values(), *(t for s in SYSTEMS.values() for t in s.values())]
                for triple in triples for e in triple if isinstance(e, str)}
    (data / "embeddings.json").write_text(json.dumps({e: vector(e) for e in elements}), encoding="utf-8")
    return data


def write_config(tmp_path: Path, data: Path, endpoint: str, metrics: dict, *,
                 systems=("alpha", "beta"), extra_general: str = "", extra_dataset: str = "") -> Path:
    metric_lines = "\n".join(f"{k} = {str(v).lower()}" for k, v in metrics.items())
    text = f"""
[general]
results_dir = "{tmp_path / 'results'}"
stats_dir = "{tmp_path / 'results' / 'stats'}"
topical_dir = "{ROOT / 'topical'}"
llm_endpoint = "{endpoint}"
llm_temperature = 0.1
max_concurrent_systems = 2
max_llm_concurrency = 4
save_stats_separate = false
compact_results = true
{extra_general}

[embeddings]
path = "{data / 'embeddings.json'}"
save_new = false

[metrics]
{metric_lines}

[datasets.toy]
display_name = "Toy"
ground_truth = "{data / 'toy_processed.json'}"
extractions_dir = "{data}"
extraction_template = "toy_{{system}}.json"
lda_model = "wire57"
systems = {json.dumps(list(systems))}
{extra_dataset}
"""
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


ALL_OFF = {m: False for m in ("TS", "US", "CS", "FS", "GS", "Coherence", "SS", "Entailment", "MINE")}
