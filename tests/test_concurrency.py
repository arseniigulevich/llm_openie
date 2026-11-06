"""gather_bounded: order, the bound on live tasks, and cancellation on failure."""

import asyncio
import random

import pytest

from evaluation.concurrency import gather_bounded


def test_results_in_input_order_and_window_respected():
    active = max_active = 0

    def factory(i):
        async def run():
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(random.random() / 200)
            active -= 1
            return i * i
        return run

    results = asyncio.run(gather_bounded([factory(i) for i in range(50)], window=5))
    assert results == [i * i for i in range(50)]
    assert max_active <= 5


def test_empty_input():
    assert asyncio.run(gather_bounded([])) == []


def test_failure_cancels_running_tasks_and_starts_no_more():
    started, cancelled = [], []

    def factory(i):
        async def run():
            started.append(i)
            if i == 2:
                await asyncio.sleep(0.01)
                raise ValueError("boom")
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled.append(i)
                raise
        return run

    async def main():
        with pytest.raises(ValueError, match="boom"):
            await gather_bounded([factory(i) for i in range(100)], window=4)
        # nothing is left running on the loop
        assert [t for t in asyncio.all_tasks() if t is not asyncio.current_task()] == []

    asyncio.run(main())
    assert sorted(started) == [0, 1, 2, 3]        # no task beyond the window was created
    assert sorted(cancelled) == [0, 1, 3]


def test_shared_semaphore_bounds_requests_across_gathers():
    sem = asyncio.Semaphore(3)
    active = max_active = 0

    async def request():
        nonlocal active, max_active
        async with sem:
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.002)
            active -= 1

    async def main():
        await asyncio.gather(*(gather_bounded([request] * 20, window=20) for _ in range(3)))

    asyncio.run(main())
    assert max_active == 3
