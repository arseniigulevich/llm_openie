"""FS, GS and Coherence with the stub endpoint: scores, empty extraction sets, order."""

import asyncio
import json
import math

import pytest

from evaluation import llm_judges
from evaluation.llm_judges import Judge
from tests import llm_stub

DATA = {
    "text one": [["a", "r", "b"], ["c", "r", "d"], ["e", "r", "f"]],
    "text two": [["g", "r", "h"]],
    "empty": [],
}


@pytest.fixture
def stub_endpoint(monkeypatch):
    calls = []

    async def stub(prompt, endpoint, model, semaphore, max_tokens=500, temperature=0.3):
        calls.append(prompt)
        await asyncio.sleep(0.001 * (len(calls) % 3))  # finish out of order
        return llm_stub.answer(prompt)

    monkeypatch.setattr(llm_judges, "hu_instruct_async", stub)
    return calls


def _judge():
    return Judge("http://unused/", "m", asyncio.Semaphore(2), temperature=0.1, window=2)


def _expected(label):
    per_text = {t: [label(json.dumps(x)) for x in ts] for t, ts in DATA.items() if ts}
    return {t: sum(v) / len(v) for t, v in per_text.items()}, per_text


def test_factualness(stub_endpoint):
    fs, details = asyncio.run(llm_judges.factualness(DATA, _judge()))
    expected, labels = _expected(llm_stub.fs_label)
    assert set(details) == set(expected)  # a text without triples gets no FS
    for text, score in expected.items():
        assert details[text]["score"] == pytest.approx(score)
        assert details[text]["results"] == labels[text]  # labels stay in triple order
    assert fs == pytest.approx(sum(expected.values()) / len(expected))
    assert len(stub_endpoint) == 4
    assert all("$TEXT$" not in p and "$TRIPLE$" not in p for p in stub_endpoint)


def test_coherence_counts_empty_sets_as_zero(stub_endpoint):
    coh, details = asyncio.run(llm_judges.coherence(DATA, _judge()))
    expected, _ = _expected(llm_stub.coherence_label)
    assert details["empty"] == {"score": 0.0, "results": []}
    assert coh == pytest.approx((sum(expected.values()) + 0.0) / 3)


def test_granularity(stub_endpoint):
    gs, answers, text2gs = asyncio.run(llm_judges.granularity(DATA, _judge()))
    for text, triples in DATA.items():
        if not triples:
            assert text not in text2gs
            continue
        expected = [math.exp(-llm_stub.gs_split(json.dumps(t))) for t in triples]
        assert text2gs[text] == pytest.approx(sum(expected) / len(expected))
        assert set(answers[text]) == {json.dumps(t) for t in triples}
    assert gs == pytest.approx(sum(text2gs.values()) / len(text2gs))


def test_granularity_scores_duplicate_triples_once(stub_endpoint):
    data = {"t": [["a", "r", "b"], ["a", "r", "b"], ["x", "r", "y"]]}
    _, answers, text2gs = asyncio.run(llm_judges.granularity(data, _judge()))
    assert len(answers["t"]) == 2
    expected = [math.exp(-llm_stub.gs_split(json.dumps(t))) for t in (["a", "r", "b"], ["x", "r", "y"])]
    assert text2gs["t"] == pytest.approx(sum(expected) / 2)


def test_no_triples_at_all(stub_endpoint):
    assert asyncio.run(llm_judges.factualness({"e": []}, _judge())) == (0.0, {})
    gs, _, text2gs = asyncio.run(llm_judges.granularity({"e": []}, _judge()))
    assert (gs, text2gs) == (1.0, {})
    assert stub_endpoint == []


def test_judge_failure_propagates(monkeypatch):
    async def failing(prompt, *a, **k):
        raise RuntimeError("endpoint down")

    monkeypatch.setattr(llm_judges, "hu_instruct_async", failing)
    with pytest.raises(RuntimeError, match="endpoint down"):
        asyncio.run(llm_judges.factualness(DATA, _judge()))
