#!/usr/bin/env python3
"""Strictly stage Static-RNN train/validation arrays on node-local storage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import sys
from pathlib import Path


REQUIRED_FILES = ("X_train.npy", "y_train.npy", "X_val.npy", "y_val.npy")
ERROR_PREFIX = "[Local-Cache-Preflight] ERROR:"
OK_PREFIX = "[Local-Cache-Preflight] OK:"


def cache_path(src_path: Path, cache_dir: Path, cache_id: str) -> Path:
    absolute_source = os.path.abspath(str(src_path))
    digest = hashlib.md5(
        f"{cache_id}_{absolute_source}".encode()
    ).hexdigest()[:8]
    stem, suffix = os.path.splitext(src_path.name)
    return cache_dir / f"{stem}_{digest}{suffix}"


def stage_dataset(
    data_dir: Path,
    cache_dir: Path,
    cache_id: str,
    reserve_bytes: int,
) -> dict[str, object]:
    if not cache_id.strip():
        raise ValueError("cache_id must be non-empty")
    data_dir = data_dir.expanduser()
    cache_dir = cache_dir.expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)

    pairs: list[tuple[Path, Path, int]] = []
    copy_bytes = 0
    for name in REQUIRED_FILES:
        source = data_dir / name
        if not source.is_file():
            raise FileNotFoundError(source)
        size = source.stat().st_size
        if size <= 0:
            raise ValueError(f"empty source file: {source}")
        target = cache_path(source, cache_dir, cache_id)
        pairs.append((source, target, size))
        if not target.is_file() or target.stat().st_size != size:
            copy_bytes += size

    free_bytes = shutil.disk_usage(cache_dir).free
    if free_bytes < copy_bytes + reserve_bytes:
        raise OSError(
            f"insufficient local space on {cache_dir}: free={free_bytes}, "
            f"copy_required={copy_bytes}, reserve={reserve_bytes}"
        )

    copied = 0
    reused = 0
    for source, target, size in pairs:
        if target.is_file() and target.stat().st_size == size:
            reused += 1
            continue
        for stale in cache_dir.glob(f"{target.name}.tmp.*"):
            if stale.is_file():
                stale.unlink()
        temporary = target.with_name(f"{target.name}.tmp.{os.getpid()}")
        try:
            shutil.copyfile(source, temporary)
            if temporary.stat().st_size != size:
                raise OSError(
                    f"copy size mismatch for {source}: {temporary.stat().st_size} != {size}"
                )
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()
        if target.stat().st_size != size:
            raise OSError(f"final cache size mismatch: {target}")
        copied += 1

    return {
        "host": socket.gethostname(),
        "data_dir": os.path.abspath(str(data_dir)),
        "cache_dir": os.path.abspath(str(cache_dir)),
        "cache_id": cache_id,
        "copied_files": copied,
        "reused_files": reused,
        "required_files": len(REQUIRED_FILES),
        "copy_bytes": copy_bytes,
        "free_bytes_before": free_bytes,
        "status": "passed",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--cache-id", required=True)
    parser.add_argument(
        "--reserve-bytes",
        type=int,
        default=1024 * 1024 * 1024,
        help="Free-space reserve retained after staging (default: 1 GiB).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = stage_dataset(
            Path(args.data_dir),
            Path(args.cache_dir),
            args.cache_id,
            args.reserve_bytes,
        )
    except Exception as exc:
        print(
            f"{ERROR_PREFIX} host={socket.gethostname()} "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 2
    print(f"{OK_PREFIX} {json.dumps(report, sort_keys=True)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
