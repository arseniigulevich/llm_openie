"""Bounded concurrent execution of many coroutines (one per LLM request)."""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, List, Optional, Sequence, TypeVar

from tqdm import tqdm

T = TypeVar("T")

# Upper bound on live tasks. The number of requests in flight is bounded separately by
# the shared LLM semaphore; this bound only keeps memory flat for large datasets.
DEFAULT_WINDOW = 1000


async def gather_bounded(
    factories: Sequence[Callable[[], Awaitable[T]]],
    *,
    window: int = DEFAULT_WINDOW,
    desc: Optional[str] = None,
) -> List[T]:
    """Run ``factory()`` for every factory, at most ``window`` at a time.

    Returns the results in input order. Coroutines are created only when a slot is free,
    so a failure leaves no coroutine un-awaited: on the first exception all running tasks
    are cancelled and awaited, and the exception is re-raised. A slow request delays only
    its own slot, not a whole chunk.
    """
    total = len(factories)
    if total == 0:
        return []
    window = max(1, window)
    results: List[Optional[T]] = [None] * total
    pending: dict = {}
    next_idx = 0
    bar = tqdm(total=total, desc=desc, disable=desc is None)

    def _launch() -> None:
        nonlocal next_idx
        while next_idx < total and len(pending) < window:
            task = asyncio.ensure_future(factories[next_idx]())
            pending[task] = next_idx
            next_idx += 1

    try:
        _launch()
        while pending:
            done, _ = await asyncio.wait(pending.keys(), return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                idx = pending.pop(task)
                results[idx] = task.result()  # re-raises the task's exception
                bar.update(1)
            _launch()
    except BaseException:
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending.keys(), return_exceptions=True)
        raise
    finally:
        bar.close()
    return results  # type: ignore[return-value]
