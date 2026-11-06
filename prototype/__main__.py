"""CLI entry point: ``python -m prototype <config.toml>``."""

from __future__ import annotations

import asyncio
import sys

from .pipeline import PipelineConfig, run_pipeline


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python -m prototype <config.toml>", file=sys.stderr)
        raise SystemExit(2)
    cfg = PipelineConfig.from_toml(sys.argv[1])
    asyncio.run(run_pipeline(cfg))


if __name__ == "__main__":
    main()
