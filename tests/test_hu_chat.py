"""The async chat client against the stub endpoint: model lookup, retries, semaphore."""

import asyncio

import pytest

from source import hu_chat
from tests.llm_stub import StubLLMServer, MODEL_ID


def test_model_is_resolved_and_answer_returned(llm_server):
    prompt = 'Relationship: ["a", "r", "b"]\nCoherence:'
    out = asyncio.run(hu_chat.hu_instruct_async(prompt, llm_server.url))
    assert out in ("coherent", "Incoherent.")
    assert hu_chat._model_cache[llm_server.url] == MODEL_ID


def test_retry_releases_semaphore_during_backoff(monkeypatch):
    monkeypatch.setattr(hu_chat, "RETRY_SLEEP_S", 0.3)
    with StubLLMServer(fail_first=1) as server:
        async def main():
            sem = asyncio.Semaphore(1)
            task = asyncio.create_task(
                hu_chat.hu_instruct_async('Triple: ["a"]\nSub-triples:', server.url, MODEL_ID, sem))
            await asyncio.sleep(0.15)               # inside the back-off after the failure
            free_during_backoff = not sem.locked()
            result = await task
            return free_during_backoff, result

        free, result = asyncio.run(main())
    assert free
    assert "Granularity" in result
    assert server.requests == 2


def test_gives_up_after_ten_attempts(monkeypatch):
    monkeypatch.setattr(hu_chat, "RETRY_SLEEP_S", 0)
    with StubLLMServer(fail_first=100) as server:
        with pytest.raises(RuntimeError, match="10 consecutive retries"):
            asyncio.run(hu_chat.hu_instruct_async("x", server.url, MODEL_ID))
    assert server.requests == 10
