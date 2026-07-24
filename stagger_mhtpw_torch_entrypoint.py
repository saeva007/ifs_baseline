#!/usr/bin/env python3
"""Stagger per-node worker startup before importing the real training module."""

from __future__ import annotations

import os
import sys
import time


def main() -> int:
    if len(sys.argv) < 2:
        print(
            "usage: stagger_mhtpw_torch_entrypoint.py TRAIN_SCRIPT [ARGS...]",
            file=sys.stderr,
        )
        return 2
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    stagger_seconds = float(os.environ.get("MHTPW_RANK_STAGGER_SECONDS", "5"))
    if stagger_seconds < 0:
        raise ValueError("MHTPW_RANK_STAGGER_SECONDS must be non-negative")
    delay = local_rank * stagger_seconds
    print(
        f"[mhtpw-startup] local_rank={local_rank} "
        f"torch_import_delay_seconds={delay:g}",
        flush=True,
    )
    if delay:
        time.sleep(delay)
    os.execv(
        sys.executable,
        [sys.executable, os.path.abspath(sys.argv[1]), *sys.argv[2:]],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
