#!/usr/bin/env python3
"""Minimal per-process HIP visibility/allocation probe for MHTPW jobs."""

from __future__ import annotations

import json
import os
import socket
import sys
import time


ERROR_PREFIX = "[DCU-Runtime-Preflight] ERROR:"
OK_PREFIX = "[DCU-Runtime-Preflight] OK:"


def read_text(path: str) -> str:
    try:
        return open(path, encoding="utf-8").read().strip()
    except OSError:
        return ""


def main() -> int:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", str(local_rank)))
    diagnostic = {
        "host": socket.gethostname(),
        "rank": rank,
        "local_rank": local_rank,
        "rocr_visible_devices": os.environ.get("ROCR_VISIBLE_DEVICES", ""),
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES", ""),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "cgroup_memory_current": read_text("/sys/fs/cgroup/memory.current"),
        "cgroup_memory_max": read_text("/sys/fs/cgroup/memory.max"),
        "mem_available_kb": next(
            (
                line.split()[1]
                for line in read_text("/proc/meminfo").splitlines()
                if line.startswith("MemAvailable:")
            ),
            "",
        ),
    }
    try:
        stagger_seconds = float(
            os.environ.get("MHTPW_RANK_STAGGER_SECONDS", "5")
        )
        if stagger_seconds < 0:
            raise ValueError(
                "MHTPW_RANK_STAGGER_SECONDS must be non-negative"
            )
        diagnostic["startup_delay_seconds"] = local_rank * stagger_seconds
        if diagnostic["startup_delay_seconds"]:
            time.sleep(diagnostic["startup_delay_seconds"])
        import torch

        diagnostic["torch"] = torch.__version__
        diagnostic["torch_hip"] = str(getattr(torch.version, "hip", ""))
        diagnostic["cuda_is_available"] = bool(torch.cuda.is_available())
        diagnostic["device_count"] = int(torch.cuda.device_count())
        if diagnostic["device_count"] < 4:
            raise RuntimeError(
                f"expected at least 4 HIP devices, found {diagnostic['device_count']}"
            )
        torch.cuda.set_device(local_rank)
        tensor = torch.ones(1024, device=f"cuda:{local_rank}")
        diagnostic["tensor_sum"] = float(tensor.sum().item())
        torch.cuda.synchronize()
    except Exception as exc:
        print(
            f"{ERROR_PREFIX} {type(exc).__name__}: {exc}; "
            f"diagnostic={json.dumps(diagnostic, sort_keys=True)}",
            file=sys.stderr,
            flush=True,
        )
        return 2
    if local_rank == 0:
        print(f"{OK_PREFIX} {json.dumps(diagnostic, sort_keys=True)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
